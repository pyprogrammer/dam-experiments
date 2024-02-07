use dam::context_tools::*;

use super::BroadcastSender;

pub struct MapTimings {
    pub initiation_interval: u64,
    pub latency: u64,
}

#[context_macro]
pub struct Map<InT: DAMType, OutT: DAMType, MapF> {
    input: Vec<Receiver<InT>>,
    output: BroadcastSender<OutT>,
    mapf: MapF,
    timings: MapTimings,
}

impl<InT: DAMType, OutT: DAMType, MapF> Map<InT, OutT, MapF>
where
    Self: Context,
{
    pub fn new(
        input: Vec<Receiver<InT>>,
        output: BroadcastSender<OutT>,
        mapf: MapF,
        timings: MapTimings,
    ) -> Self {
        let s = Self {
            input,
            output,
            mapf,
            timings,
            context_info: Default::default(),
        };
        s.input.iter().for_each(|chn| chn.attach_receiver(&s));
        s.output.attach_sender(&s);
        s
    }
}

impl<InT: DAMType, OutT: DAMType, MapF> Context for Map<InT, OutT, MapF>
where
    MapF: Fn(&[InT]) -> OutT + Sync + Send,
{
    fn run(&mut self) {
        loop {
            // Block on all of the inputs
            self.input.iter().for_each(|chn| {
                let _ = chn.peek_next(&self.time);
            });
            let dequeued: Vec<_> = self
                .input
                .iter()
                .map(|chn| chn.dequeue(&self.time))
                .collect();
            if dequeued.iter().any(|v| v.is_err()) {
                return;
            }
            let data: Vec<_> = dequeued.into_iter().map(|v| v.unwrap().data).collect();
            let output = (self.mapf)(&data);
            self.output
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick() + self.timings.latency,
                        data: output,
                    },
                )
                .unwrap();
            self.time.incr_cycles(self.timings.initiation_interval);
        }
    }
}

#[cfg(test)]
mod tests {
    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{CheckerContext, GeneratorContext},
    };

    use crate::templates::BroadcastSender;

    use super::{Map, MapTimings};

    #[test]
    fn test_map() {
        let mut builder = ProgramBuilder::default();
        let (in_snd, in_rcv) = builder.bounded(16);
        let (out_snd, out_rcv) = builder.bounded(16);
        builder.add_child(GeneratorContext::new(|| (0..16), in_snd));
        builder.add_child(CheckerContext::new(|| (0..16).map(|x| x + 5), out_rcv));
        builder.add_child(Map::new(
            vec![in_rcv],
            BroadcastSender {
                targets: vec![out_snd],
            },
            |v| v[0] + 5,
            MapTimings {
                initiation_interval: 1,
                latency: 5,
            },
        ));
        let elapsed = builder
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .elapsed_cycles();
        dbg!(elapsed);
    }
}
