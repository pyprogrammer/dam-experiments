use dam::context_tools::*;

use super::BroadcastSender;

pub struct ScanTimings {
    pub initiation_interval: u64,
    pub latency: u64,
}

#[context_macro]
pub struct Scan<InT: DAMType, OutT: DAMType, UpdateT> {
    reset_freq: usize,
    input: Receiver<InT>,
    output: BroadcastSender<OutT>,
    update_fn: UpdateT,
    timings: ScanTimings,
}

impl<InT: DAMType, OutT: DAMType, UpdateT> Scan<InT, OutT, UpdateT>
where
    Self: Context,
{
    pub fn new(
        reset_freq: usize,
        input: Receiver<InT>,
        output: BroadcastSender<OutT>,
        update_fn: UpdateT,
        timings: ScanTimings,
    ) -> Self {
        let s = Self {
            reset_freq,
            input,
            output,
            update_fn,
            timings,
            context_info: Default::default(),
        };
        s.input.attach_receiver(&s);
        s.output.attach_sender(&s);
        s
    }
}

impl<InT: DAMType, OutT: DAMType, UpdateT> Context for Scan<InT, OutT, UpdateT>
where
    UpdateT: Sync + Send + Fn(InT, Option<&OutT>) -> OutT,
{
    fn run(&mut self) {
        // Infinite loop to handle all inputs
        loop {
            let mut accum: Option<OutT> = None;
            for iter in 0..self.reset_freq {
                let input = match self.input.dequeue(&self.time) {
                    Ok(ChannelElement { time: _, data }) => data,
                    Err(_) if iter == 0 => return,
                    Err(_) => panic!(
                        "Premature End of Receiver {:?} on Scan {:?}",
                        self.input.id(),
                        self.id
                    ),
                };
                let new_val = (self.update_fn)(input, accum.as_ref());
                self.output
                    .enqueue(
                        &self.time,
                        ChannelElement {
                            time: self.time.tick() + self.timings.latency,
                            data: new_val.clone(),
                        },
                    )
                    .unwrap_or_else(|_| panic!("Premature End of Sender on Scan {:?}", self.id));
                accum = Some(new_val);
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

    use super::Scan;

    #[test]
    fn scan_test() {
        let mut builder = ProgramBuilder::default();
        let values: Vec<Vec<u64>> = vec![
            (0..10).collect(),
            (10..20).collect(),
            (20..30).collect(),
            (30..40).collect(),
        ];
        let inputs: Vec<_> = values.iter().flat_map(|x| x.iter().copied()).collect();
        let (in_snd, in_rcv) = builder.bounded(16);
        builder.add_child(GeneratorContext::new(|| inputs.into_iter(), in_snd));

        let (out_snd, out_rcv) = builder.bounded(16);
        builder.add_child(Scan::new(
            10,
            in_rcv,
            BroadcastSender {
                targets: vec![out_snd],
            },
            |new, old| match old {
                Some(old_val) => new + old_val,
                None => new,
            },
            super::ScanTimings {
                initiation_interval: 2,
                latency: 1,
            },
        ));
        let gold: Vec<_> = values
            .iter()
            .flat_map(|vals| {
                let mut cumsums = vec![];
                for (i, v) in vals.iter().enumerate() {
                    if i == 0 {
                        cumsums.push(*v);
                    } else {
                        cumsums.push(cumsums.last().unwrap() + *v);
                    }
                }
                cumsums.into_iter()
            })
            .collect();
        builder.add_child(CheckerContext::new(|| gold.into_iter(), out_rcv));
        let elapsed = builder
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .elapsed_cycles();
        dbg!(elapsed);
    }
}
