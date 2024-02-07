use dam::{context_tools::*, structures::SyncSendMarker};

use super::BroadcastSender;

pub struct FlatmapTimings {
    pub initiation_interval: u64,
    pub latency: u64,
}

#[context_macro]
pub struct Flatmap<InT: DAMType, OutT: DAMType, FlatmapF, IType> {
    input: Vec<Receiver<InT>>,
    output: BroadcastSender<OutT>,
    flatmapf: FlatmapF,
    timings: FlatmapTimings,

    _marker: SyncSendMarker<IType>,
}

impl<InT: DAMType, OutT: DAMType, FlatmapF, IType> Flatmap<InT, OutT, FlatmapF, IType>
where
    Self: Context,
{
    pub fn new(
        input: Vec<Receiver<InT>>,
        output: BroadcastSender<OutT>,
        flatmapf: FlatmapF,
        timings: FlatmapTimings,
    ) -> Self {
        let s = Self {
            input,
            output,
            flatmapf,
            timings,
            _marker: Default::default(),
            context_info: Default::default(),
        };
        s.input.iter().for_each(|chn| chn.attach_receiver(&s));
        s.output.attach_sender(&s);
        s
    }
}

impl<InT: DAMType, OutT: DAMType, FlatmapF, IType> Context for Flatmap<InT, OutT, FlatmapF, IType>
where
    FlatmapF: (Fn(Vec<InT>) -> IType) + Sync + Send,
    IType: Iterator<Item = OutT>,
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
            let outputs = (self.flatmapf)(data);
            for output in outputs {
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
}

#[cfg(test)]
mod tests {
    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{CheckerContext, GeneratorContext},
    };

    use crate::templates::BroadcastSender;

    use super::{Flatmap, FlatmapTimings};

    #[test]
    fn test_flatmap() {
        let mut builder = ProgramBuilder::default();
        let (in_snd, in_rcv) = builder.bounded(16);
        let (out_snd, out_rcv) = builder.bounded(16);
        builder.add_child(GeneratorContext::new(|| (0..16), in_snd));
        builder.add_child(CheckerContext::new(|| (0..16).flat_map(|x| 0..x), out_rcv));
        builder.add_child(Flatmap::new(
            vec![in_rcv],
            BroadcastSender {
                targets: vec![out_snd],
            },
            |v| (0..v[0]),
            FlatmapTimings {
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
