use dam::context_tools::*;

pub struct ReduceTimings {
    pub initiation_interval: u64,
    pub latency: u64,
    pub reset_time: u64,
}

#[context_macro]
pub struct Reduce<InT: DAMType, OutT: DAMType, UpdateT> {
    reset_freq: usize,
    input: Receiver<InT>,
    output: Sender<OutT>,
    update_fn: UpdateT,
    timings: ReduceTimings,
}

impl<InT: DAMType, OutT: DAMType, UpdateT> Reduce<InT, OutT, UpdateT>
where
    Self: Context,
{
    pub fn new(
        reset_freq: usize,
        input: Receiver<InT>,
        output: Sender<OutT>,
        update_fn: UpdateT,
        timings: ReduceTimings,
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

impl<InT: DAMType, OutT: DAMType, UpdateT> Context for Reduce<InT, OutT, UpdateT>
where
    UpdateT: Sync + Send + Fn(InT, Option<OutT>) -> OutT,
{
    fn run(&mut self) {
        // Infinite loop to handle all inputs
        loop {
            self.time.incr_cycles(self.timings.reset_time);
            let mut accum: Option<OutT> = None;
            for iter in 0..self.reset_freq {
                let input = match self.input.dequeue(&self.time) {
                    Ok(ChannelElement { time: _, data }) => data,
                    Err(_) if iter == 0 => return,
                    Err(_) => panic!(
                        "Premature End of Receiver {:?} on Reduce {:?}",
                        self.input.id(),
                        self.id
                    ),
                };
                let new_val = (self.update_fn)(input, accum);
                accum = Some(new_val);
                self.time.incr_cycles(self.timings.initiation_interval);
            }
            self.output
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick() + self.timings.latency,
                        data: accum.unwrap(),
                    },
                )
                .unwrap_or_else(|_| {
                    panic!(
                        "Premature End of Sender {:?} on Reduce {:?}",
                        self.output.id(),
                        self.id
                    )
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{CheckerContext, GeneratorContext},
    };

    use super::Reduce;

    #[test]
    fn reduce_test() {
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
        builder.add_child(Reduce::new(
            10,
            in_rcv,
            out_snd,
            |new, old| match old {
                Some(old_val) => new + old_val,
                None => new,
            },
            super::ReduceTimings {
                initiation_interval: 2,
                latency: 1,
                reset_time: 0,
            },
        ));
        let gold: Vec<_> = values.iter().map(|x| x.iter().sum()).collect();
        builder.add_child(CheckerContext::new(|| gold.into_iter(), out_rcv));
        let elapsed = builder
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .elapsed_cycles();
        dbg!(elapsed);
    }
}
