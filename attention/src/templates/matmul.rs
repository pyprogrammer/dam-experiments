use dam::context_tools::*;

#[derive(Debug, Copy, Clone)]
pub struct MatmulTiming {
    pub dot_latency: u64,
    pub dot_ii: u64,
    pub reset_time: u64,
}

#[derive(Debug, Copy, Clone)]
pub struct ShapeInfo {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum MatmulBehavior {
    Buffered,
    Repeated,
}

/// Computes A: [M, K] x B[K, N] = C [M, N]
/// Options:
/// 1. The K dimension of A is buffered, so it reads it once.
/// 2. The K dimension of A is repeated, so it reads it once per iteration (repeated M times)
#[context_macro]
pub struct Matmul<InputT, OutputT, MacT>
where
    InputT: DAMType,
    OutputT: DAMType,
{
    timing: MatmulTiming,
    behavior: MatmulBehavior,
    shape: ShapeInfo,
    left: Receiver<InputT>,
    right: Receiver<InputT>,
    output: Sender<OutputT>,
    mac: MacT,
}

impl<InputT, OutputT, MacT> Matmul<InputT, OutputT, MacT>
where
    InputT: DAMType,
    OutputT: DAMType + num::Float,
    MacT: Fn(InputT, InputT, OutputT) -> OutputT + Sync + Send,
{
    pub fn new(
        timing: MatmulTiming,
        behavior: MatmulBehavior,
        shape: ShapeInfo,
        left: Receiver<InputT>,
        right: Receiver<InputT>,
        output: Sender<OutputT>,
        mac: MacT,
    ) -> Self {
        let output = Self {
            timing,
            behavior,
            shape,
            left,
            right,
            output,
            mac,
            context_info: Default::default(),
        };
        output.left.attach_receiver(&output);
        output.right.attach_receiver(&output);
        output.output.attach_sender(&output);
        output
    }

    fn buffered_matmul(&self) {
        let mut left_buffer = Vec::with_capacity(self.shape.k);
        loop {
            // Loop over M
            for m in 0..self.shape.m {
                for n in 0..self.shape.n {
                    let should_populate_buffer = n == 0;
                    let mut accum = OutputT::zero();
                    for k in 0..self.shape.k {
                        // Align the two timings
                        let right_peek = self.right.peek_next(&self.time);
                        if right_peek.is_err() {
                            if m == 0 && n == 0 && k == 0 {
                                return;
                            }
                            panic!("Unexpected termination of right stream in matmul ID: {:?} at time {:?} on iteration {m}, {n}, {k}", self.id, self.time.tick());
                        }
                        if should_populate_buffer {
                            match self.left.dequeue(&self.time) {
                                Ok(ChannelElement { time: _, data }) => left_buffer.push(data),
                                Err(_) if m == 0 && n == 0 && k == 0 => return,
                                Err(_) => {
                                    panic!("Unexpected termination of left stream in matmul ID: {:?} at time {:?} on iteration {m}, {n}, {k}", self.id, self.time.tick());
                                }
                            }
                        }
                        let ChannelElement {
                            time: _,
                            data: right_data,
                        } = self.right.dequeue(&self.time).unwrap();
                        let left_data = left_buffer[k].clone();
                        accum = (self.mac)(left_data, right_data, accum);
                        self.time.incr_cycles(self.timing.dot_ii);
                    }
                    // After K values, we spit out the accum.
                    self.output
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick() + self.timing.dot_latency,
                                data: accum,
                            },
                        )
                        .unwrap_or_else(|_| {
                            panic!(
                                "Unexpected termination of output channel on Matmul {:?}",
                                self.id
                            )
                        });
                }
                // Reset buffer after N elements as we prepare to read the next inputs.
                left_buffer.clear();
            }
        }
    }
    fn repeated_matmul(&self) {
        // For processing multiple batches
        loop {
            // Looping over M
            for m in 0..self.shape.m {
                self.time.incr_cycles(self.timing.reset_time);
                // Looping over N
                for n in 0..self.shape.n {
                    let mut accum = OutputT::zero();
                    // Looping over K (common dim)
                    for k in 0..self.shape.k {
                        let _ = self.left.peek_next(&self.time);
                        let _ = self.right.peek_next(&self.time);
                        match (
                            self.left.dequeue(&self.time),
                            self.right.dequeue(&self.time),
                        ) {
                            (
                                Ok(ChannelElement { time: _, data: a }),
                                Ok(ChannelElement { time: _, data: b }),
                            ) => {
                                // Perform a MAC
                                accum = (self.mac)(a, b, accum);
                            }
                            (_, Err(_)) | (Err(_), _) if (m == 0 && n == 0 && k == 0) => {
                                // Finished all of our iterations
                                return;
                            }
                            _ => {
                                panic!("Unexpected termination of streams in matmul ID: {:?} at time {:?} on iteration {m}, {n}, {k}", self.id, self.time.tick());
                            }
                        }

                        self.time.incr_cycles(self.timing.dot_ii);
                    }
                    self.output
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick() + self.timing.dot_latency,
                                data: accum,
                            },
                        )
                        .unwrap_or_else(|_| {
                            panic!(
                                "Unexpected termination of output channel on Matmul {:?}",
                                self.id
                            )
                        });
                }
            }
        }
    }
}

impl<InputT, OutputT, MacT> Context for Matmul<InputT, OutputT, MacT>
where
    InputT: DAMType,
    OutputT: DAMType + num::Float,
    MacT: Fn(InputT, InputT, OutputT) -> OutputT + Sync + Send,
{
    fn run(&mut self) {
        match self.behavior {
            MatmulBehavior::Buffered => self.buffered_matmul(),
            MatmulBehavior::Repeated => self.repeated_matmul(),
        }
    }
}

#[cfg(test)]
mod tests {
    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{ApproxCheckerContext, GeneratorContext},
    };
    use ndarray::ArcArray;

    use super::*;

    const CHAN_DEPTH: usize = 8;

    fn run_test(
        behavior: MatmulBehavior,
        timing: MatmulTiming,
        shape: ShapeInfo,
        outer_iterations: usize,
    ) {
        // generate the input matrices
        let a_matrices = (0..outer_iterations)
            .map(|_| ArcArray::from_shape_simple_fn([shape.m, shape.k], fastrand::f32))
            .collect::<Vec<_>>();
        let b_matrices = (0..outer_iterations)
            .map(|_| ArcArray::from_shape_simple_fn([shape.k, shape.n], fastrand::f32))
            .collect::<Vec<_>>();

        let mut builder = ProgramBuilder::default();
        let (a_snd, a_recv) = builder.bounded(CHAN_DEPTH);
        let (b_snd, b_recv) = builder.bounded(CHAN_DEPTH);
        let (c_snd, c_recv) = builder.bounded(CHAN_DEPTH);

        match behavior {
            MatmulBehavior::Buffered => {
                // Simpler case because we can write A as-is
                builder.add_child(GeneratorContext::new(
                    || a_matrices.iter().flat_map(|mat| mat.into_iter()).copied(),
                    a_snd,
                ));
            }
            MatmulBehavior::Repeated => {
                // More complex case because we need to write each row of A repeatedly
                builder.add_child(GeneratorContext::new(
                    || {
                        a_matrices.iter().flat_map(|mat| {
                            mat.rows()
                                .into_iter()
                                .flat_map(|row| {
                                    // repeat the row N times
                                    (0..shape.n).flat_map(move |_| row.into_iter())
                                })
                                .copied()
                        })
                    },
                    a_snd,
                ));
            }
        }

        builder.add_child(GeneratorContext::new(
            || {
                b_matrices.iter().flat_map(|mat_b| {
                    (0..shape.m).flat_map(move |_| {
                        mat_b.t().iter().copied().collect::<Vec<_>>().into_iter()
                    })
                })
            },
            b_snd,
        ));
        // The matmul node
        builder.add_child(Matmul::new(
            timing,
            behavior,
            shape,
            a_recv,
            b_recv,
            c_snd,
            |a, b, c| a * b + c,
        ));

        builder.add_child(ApproxCheckerContext::new(
            || {
                a_matrices.iter().zip(b_matrices.iter()).flat_map(|(a, b)| {
                    let gold = a.dot(b);
                    gold.into_iter()
                })
            },
            c_recv,
            |a, b| (a - b).abs() < 0.001,
        ));

        let executed = builder
            .initialize(Default::default())
            .unwrap()
            .run(Default::default());
        dbg!(executed.elapsed_cycles());
    }

    #[test]
    fn run_buffered() {
        run_test(
            MatmulBehavior::Buffered,
            MatmulTiming {
                dot_latency: 1,
                dot_ii: 1,
                reset_time: 0,
            },
            ShapeInfo {
                m: 512,
                n: 32,
                k: 16,
            },
            4,
        );
    }

    #[test]
    fn run_repeated() {
        run_test(
            MatmulBehavior::Repeated,
            MatmulTiming {
                dot_latency: 1,
                dot_ii: 1,
                reset_time: 0,
            },
            ShapeInfo {
                m: 512,
                n: 32,
                k: 16,
            },
            4,
        );
    }
}
