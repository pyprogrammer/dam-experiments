use ndarray::{Array2, ArrayView2, Axis};

pub mod agnostic;
pub mod naive;

#[derive(Clone, Copy, Debug)]
pub struct AttentionConfig {
    pub vocab_dim: usize,
    pub seq_len: usize,
}

pub fn compute_attention<T: num::Float + std::fmt::Debug + 'static>(
    q: ArrayView2<T>,
    k: ArrayView2<T>,
    v: ArrayView2<T>,
) -> Array2<T> {
    let qk_transpose = q.dot(&k.t());
    let row_max = qk_transpose.fold_axis(Axis(1), T::min_value(), |x, y| x.max(*y));
    let normalized = qk_transpose - row_max.into_shape((q.nrows(), 1usize)).unwrap();
    let exponentiated = normalized.map(|x| x.exp());
    let row_sum = exponentiated
        .sum_axis(Axis(1))
        .into_shape((q.nrows(), 1usize))
        .unwrap();
    let divided = exponentiated / row_sum;
    divided.dot(&v)
}

#[cfg(test)]
mod tests {
    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{ApproxCheckerContext, GeneratorContext},
    };
    use ndarray::ArcArray;

    use crate::{
        apps::{
            agnostic::{agnostic_attention, AgnosticConfig},
            compute_attention, AttentionConfig,
        },
        templates::{MapTimings, Matmul, MatmulTiming, ReduceTimings, ScanTimings, ShapeInfo},
        FlatmapTimings,
    };

    use super::naive;

    #[test]
    fn test_naive_attention() {
        const SEQ_LEN: usize = 256;
        const DIM: usize = 4;
        const SHORT_DEPTH: usize = 16;
        const LONG_DEPTH: usize = SEQ_LEN + 2;
        let q = ArcArray::from_shape_simple_fn([SEQ_LEN, DIM], fastrand::f64);
        let k = ArcArray::from_shape_simple_fn([SEQ_LEN, DIM], fastrand::f64);
        let v = ArcArray::from_shape_simple_fn([SEQ_LEN, DIM], fastrand::f64);
        let attn = compute_attention(q.view(), k.view(), v.view());

        // dbg!(&q);
        // dbg!(&k);
        // dbg!(&v);
        // dbg!(&attn);

        let mut builder = ProgramBuilder::default();

        // Assemble the matmul
        let qkt_receiver = {
            let (a_snd, a_recv) = builder.bounded(SHORT_DEPTH);
            let (b_snd, b_recv) = builder.bounded(SHORT_DEPTH);
            let (qkt_sender, qkt_receiver) = builder.bounded(SHORT_DEPTH);

            builder.add_child(GeneratorContext::new(|| q.into_iter(), a_snd));
            builder.add_child(GeneratorContext::new(
                || {
                    (0..SEQ_LEN)
                        .flat_map(move |_| k.iter().copied().collect::<Vec<_>>().into_iter())
                },
                b_snd,
            ));

            builder.add_child(Matmul::new(
                MatmulTiming {
                    dot_latency: 1,
                    dot_ii: 1,
                },
                crate::templates::MatmulBehavior::Buffered,
                ShapeInfo {
                    m: SEQ_LEN,
                    n: SEQ_LEN,
                    k: DIM,
                },
                a_recv,
                b_recv,
                qkt_sender,
                |a, b, c: f64| a * b + c,
            ));

            qkt_receiver
        };

        // builder.add_child(PrinterContext::new(qkt_receiver));

        let (v_snd, v_recv) = builder.bounded(SHORT_DEPTH);
        builder.add_child(GeneratorContext::new(
            || {
                (0..SEQ_LEN)
                    .flat_map(move |_| v.t().iter().copied().collect::<Vec<_>>().into_iter())
            },
            v_snd,
        ));

        let naive_attn = naive::naive(
            &mut builder,
            qkt_receiver,
            v_recv,
            AttentionConfig {
                vocab_dim: DIM,
                seq_len: SEQ_LEN,
            },
            naive::NaiveConfig {
                long_chan_size: LONG_DEPTH,
                short_chan_depth: SHORT_DEPTH,
                exp_timings: MapTimings {
                    initiation_interval: 1,
                    latency: 1,
                },
                div_timings: MapTimings {
                    initiation_interval: 1,
                    latency: 1,
                },
                sum_timings: ReduceTimings {
                    initiation_interval: 1,
                    latency: 1,
                },
                matmul_timings: MatmulTiming {
                    dot_latency: 1,
                    dot_ii: 1,
                },
            },
        );
        builder.add_child(ApproxCheckerContext::new(
            || attn.into_iter(),
            naive_attn,
            |a, b| (a - b).abs() < 0.01,
        ));

        let executed = builder
            .initialize(Default::default())
            .unwrap()
            .run(Default::default());
        dbg!(executed.elapsed_cycles());
    }

    #[test]
    fn test_agnostic_attention() {
        const SEQ_LEN: usize = 256;
        const DIM: usize = 4;
        const SHORT_DEPTH: usize = 16;
        let q = ArcArray::from_shape_simple_fn([SEQ_LEN, DIM], fastrand::f64);
        let k = ArcArray::from_shape_simple_fn([SEQ_LEN, DIM], fastrand::f64);
        let v = ArcArray::from_shape_simple_fn([SEQ_LEN, DIM], fastrand::f64);
        let attn = compute_attention(q.view(), k.view(), v.view());

        // dbg!(&q);
        // dbg!(&k);
        // dbg!(&v);
        // dbg!(&attn);

        let mut builder = ProgramBuilder::default();

        // Assemble the matmul
        let qkt_receiver = {
            let (a_snd, a_recv) = builder.bounded(SHORT_DEPTH);
            let (b_snd, b_recv) = builder.bounded(SHORT_DEPTH);
            let (qkt_sender, qkt_receiver) = builder.bounded(SHORT_DEPTH);

            builder.add_child(GeneratorContext::new(|| q.into_iter(), a_snd));
            builder.add_child(GeneratorContext::new(
                || {
                    (0..SEQ_LEN)
                        .flat_map(move |_| k.iter().copied().collect::<Vec<_>>().into_iter())
                },
                b_snd,
            ));

            builder.add_child(Matmul::new(
                MatmulTiming {
                    dot_latency: 1,
                    dot_ii: 1,
                },
                crate::templates::MatmulBehavior::Buffered,
                ShapeInfo {
                    m: SEQ_LEN,
                    n: SEQ_LEN,
                    k: DIM,
                },
                a_recv,
                b_recv,
                qkt_sender,
                |a, b, c: f64| a * b + c,
            ));

            qkt_receiver
        };

        // builder.add_child(PrinterContext::new(qkt_receiver));

        let (v_snd, v_recv) = builder.bounded(SHORT_DEPTH);
        builder.add_child(GeneratorContext::new(
            || (0..SEQ_LEN).flat_map(move |_| v.iter().copied().collect::<Vec<_>>().into_iter()),
            v_snd,
        ));

        let agnostic_attn = agnostic_attention(
            &mut builder,
            qkt_receiver,
            v_recv,
            AttentionConfig {
                vocab_dim: DIM,
                seq_len: SEQ_LEN,
            },
            AgnosticConfig {
                chan_depth: SHORT_DEPTH,
                max_config: ScanTimings {
                    initiation_interval: 1,
                    latency: 1,
                },
                residual_config: ReduceTimings {
                    initiation_interval: 1,
                    latency: 1,
                },
                prod_config: ReduceTimings {
                    initiation_interval: 1,
                    latency: 1,
                },
                scale_config: FlatmapTimings {
                    initiation_interval: 1,
                    latency: 1,
                },
            },
        );

        builder.add_child(ApproxCheckerContext::new(
            || attn.into_iter(),
            agnostic_attn,
            |a, b| (a - b).abs() < 0.01,
        ));
        // builder.add_child(PrinterContext::new(agnostic_attn));

        let executed = builder
            .initialize(Default::default())
            .unwrap()
            .run(Default::default());
        dbg!(executed.elapsed_cycles());
    }
}
