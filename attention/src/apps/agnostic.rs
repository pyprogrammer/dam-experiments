use dam::{context_tools::*, simulation::ProgramBuilder};

use crate::templates::*;

use super::AttentionConfig;

pub struct AgnosticConfig {
    pub chan_depth: usize,
    pub max_config: ScanTimings,
    pub residual_config: ReduceTimings,
    pub prod_config: ReduceTimings,
    pub scale_config: FlatmapTimings,
}

#[derive(Clone, Copy, Debug, Default)]
struct RunningResult<T> {
    /// m_i^(j)
    cur_max: T,

    /// m_i^(j - 1) - m_i^(j)
    delta_max: T,

    /// e^(S_ij - m_i^(j))
    exp: T,

    /// e^(m_i^(j))
    delta_elem: T,
}

impl<T: DAMType> DAMType for RunningResult<T> {
    fn dam_size(&self) -> usize {
        self.cur_max.dam_size()
            + self.delta_max.dam_size()
            + self.exp.dam_size()
            + self.delta_elem.dam_size()
    }
}

#[derive(Clone, Debug, Default)]
struct Vector<T> {
    pub value: Vec<T>,
}

impl<T: DAMType> DAMType for Vector<T> {
    fn dam_size(&self) -> usize {
        self.value.iter().map(|x| x.dam_size()).sum()
    }
}

pub fn agnostic_attention<'a, T: DAMType + num::Float>(
    builder: &mut ProgramBuilder<'a>,
    qkt_receiver: Receiver<T>,
    v_receiver: Receiver<T>,
    config: AttentionConfig,
    agnostic_config: AgnosticConfig,
) -> Receiver<T>
where
    T: 'a,
{
    let (scan_to_residual_snd, scan_to_residual_rcv) = builder.bounded(agnostic_config.chan_depth);
    let (scan_to_mul_snd, scan_to_mul_rcv) = builder.bounded(agnostic_config.chan_depth);

    builder.add_child(Scan::new(
        config.seq_len,
        qkt_receiver,
        BroadcastSender {
            targets: vec![scan_to_residual_snd, scan_to_mul_snd],
        },
        |new, old| match old {
            Some(RunningResult {
                cur_max: old_max,
                delta_max: _,
                exp: _,
                delta_elem: _,
            }) => {
                let new_max = new.max(*old_max);
                let delta_max = *old_max - new_max;

                RunningResult {
                    cur_max: new_max,
                    delta_max,
                    exp: (new - new_max).exp(),
                    delta_elem: delta_max.exp(),
                }
            }
            None => RunningResult {
                cur_max: new,
                delta_max: new,
                exp: T::one(),
                delta_elem: new.exp(),
            },
        },
        agnostic_config.max_config,
    ));

    let (r_to_div_rep_snd, r_to_div_rep_rcv) = builder.bounded(agnostic_config.chan_depth);

    builder.add_child(Reduce::new(
        config.seq_len,
        scan_to_residual_rcv,
        r_to_div_rep_snd,
        |RunningResult {
             cur_max: _,
             delta_max: _,
             exp,
             delta_elem,
         },
         old: Option<T>| match old {
            // On future iterations, r_i^(j) = r_i^(j-1) * delta_ij + e_ij
            Some(old_val) => old_val * delta_elem + exp,
            // On the first iteration, r_i^(j) is zero
            None => exp,
        },
        agnostic_config.residual_config,
    ));

    let (reduce_to_div_snd, reduce_to_div_rcv) = builder.bounded(agnostic_config.chan_depth);

    let (v_vec_snd, v_vec_rcv) = builder.bounded(agnostic_config.chan_depth);

    // Read rows of the V matrix as vectors.
    builder.add_child(Reduce::new(
        config.vocab_dim,
        v_receiver,
        v_vec_snd,
        move |new, old: Option<Vector<T>>| match old {
            Some(mut v) => {
                v.value.push(new);
                v
            }
            None => {
                let mut v = Vector {
                    value: Vec::with_capacity(config.vocab_dim),
                };
                v.value.push(new);
                v
            }
        },
        ReduceTimings {
            initiation_interval: 1,
            latency: 1,
        },
    ));

    let (mul_in_snd, mul_in_rcv) = builder.bounded(agnostic_config.chan_depth);
    builder.add_child(Zip::new(
        scan_to_mul_rcv,
        v_vec_rcv,
        BroadcastSender {
            targets: vec![mul_in_snd],
        },
    ));

    // Scale each vector by a compensating factor
    builder.add_child(Reduce::new(
        config.seq_len,
        mul_in_rcv,
        reduce_to_div_snd,
        move |Pair(
            RunningResult {
                cur_max: _,
                delta_max: _,
                exp,
                delta_elem,
            },
            mut v_vector,
        ),
              old: Option<Vector<T>>| match old {
            Some(mut old_val) => {
                for i in 0..old_val.value.len() {
                    old_val.value[i] = old_val.value[i] * delta_elem + exp * v_vector.value[i]
                }

                old_val
            }
            None => {
                v_vector.value.iter_mut().for_each(|vec_val| {
                    *vec_val = *vec_val * exp;
                });

                v_vector
            }
        },
        agnostic_config.prod_config,
    ));

    let (output_snd, output_rcv) = builder.bounded(agnostic_config.chan_depth);

    let (reduce_snd, reduce_rcv) = builder.bounded(agnostic_config.chan_depth);
    builder.add_child(Zip::new(
        reduce_to_div_rcv,
        r_to_div_rep_rcv,
        BroadcastSender {
            targets: vec![reduce_snd],
        },
    ));

    builder.add_child(Flatmap::new(
        vec![reduce_rcv],
        BroadcastSender {
            targets: vec![output_snd],
        },
        |mut inputs| {
            let Pair(vector, scale) = inputs.pop().unwrap();
            vector.value.into_iter().map(move |v| v / scale)
        },
        agnostic_config.scale_config,
    ));

    output_rcv
}
