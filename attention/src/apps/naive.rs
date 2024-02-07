use dam::{context_tools::*, simulation::ProgramBuilder};

use crate::templates::*;

use super::AttentionConfig;

pub struct NaiveConfig {
    pub long_chan_size: usize,
    pub short_chan_depth: usize,
    pub exp_timings: MapTimings,
    pub div_timings: MapTimings,
    pub sum_timings: ReduceTimings,
    pub matmul_timings: MatmulTiming,
}

pub fn naive<'a, T: DAMType + num::Float>(
    builder: &mut ProgramBuilder<'a>,
    qkt_receiver: Receiver<T>,
    v_receiver: Receiver<T>,
    config: AttentionConfig,
    naive_config: NaiveConfig,
) -> Receiver<T>
where
    T: 'a,
{
    let (exp_to_div_snd, exp_to_div_rcv) = builder.bounded(naive_config.long_chan_size);
    let (exp_to_sum_snd, exp_to_sum_rcv) = builder.bounded(naive_config.short_chan_depth);
    let (sum_to_rep_snd, sum_to_rep_rcv) = builder.bounded(naive_config.short_chan_depth);
    let (rep_to_div_snd, rep_to_div_rcv) = builder.bounded(naive_config.short_chan_depth);
    // Map over e^x
    builder.add_child(Map::new(
        vec![qkt_receiver],
        BroadcastSender {
            targets: vec![exp_to_div_snd, exp_to_sum_snd],
        },
        |qkt| qkt[0].exp(),
        naive_config.exp_timings,
    ));

    builder.add_child(Reduce::new(
        config.seq_len,
        exp_to_sum_rcv,
        sum_to_rep_snd,
        |new, cur| match cur {
            Some(x) => new + x,
            None => new,
        },
        naive_config.sum_timings,
    ));

    builder.add_child(Repeat::new(
        sum_to_rep_rcv,
        BroadcastSender {
            targets: vec![rep_to_div_snd],
        },
        config.seq_len,
    ));

    let (div_to_mm_snd, div_to_mm_rcv) = builder.bounded(naive_config.short_chan_depth);
    // Map over p_ij = e_ij / r_i for the softmax
    builder.add_child(Map::new(
        vec![exp_to_div_rcv, rep_to_div_rcv],
        BroadcastSender {
            targets: vec![div_to_mm_snd],
        },
        |args| args[0] / args[1],
        naive_config.div_timings,
    ));

    // take the product of p_ij with v to get the result.
    let (output_snd, output_rcv) = builder.bounded(naive_config.short_chan_depth);

    builder.add_child(Matmul::new(
        naive_config.matmul_timings,
        MatmulBehavior::Buffered,
        ShapeInfo {
            m: config.seq_len,
            n: config.vocab_dim,
            k: config.seq_len,
        },
        div_to_mm_rcv,
        v_receiver,
        output_snd,
        |a, b, c| (a * b) + c,
    ));

    output_rcv
}
