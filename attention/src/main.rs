pub mod apps;
pub mod templates;
pub mod utils;
use clap::{Args, Parser, Subcommand};
use dam::{
    simulation::{ProgramBuilder, RunMode, RunOptionsBuilder},
    utility_contexts::*,
};
use itertools::izip;
use ndarray::ArcArray;

use crate::{
    apps::{agnostic::AgnosticConfig, compute_attention, AttentionConfig},
    templates::*,
};

#[derive(Parser, Debug)]
struct CommandLineInterface {
    /// Naive or Memory-agnostic attention
    #[command(subcommand)]
    mode: Implementation,

    /// The sequence length (N)
    #[arg(long)]
    length: usize,

    /// The dimensionality of the tokens (D)
    #[arg(short, long)]
    dim: usize,

    /// The batch size (B)
    #[arg(short, long, default_value_t = 1)]
    batch: usize,

    #[command(flatten)]
    common: CommonTimings,

    /// Validate results afterwards
    #[arg(long, default_value_t = false)]
    validate: bool,

    /// Limit the number of worker threads
    #[arg(long)]
    workers: Option<usize>,
}

#[derive(Subcommand, Debug, Copy, Clone)]
enum Implementation {
    Naive {
        #[arg(long)]
        short_depth: usize,

        #[arg(long)]
        long_depth: usize,

        #[arg(long, default_value_t = 1)]
        exp_ii: u64,

        #[arg(long, default_value_t = 1)]
        exp_latency: u64,

        #[arg(long, default_value_t = 1)]
        sum_ii: u64,

        #[arg(long, default_value_t = 1)]
        sum_latency: u64,
    },
    Agnostic {
        #[arg(long)]
        channel_depth: usize,

        #[arg(long, default_value_t = 1)]
        max_ii: u64,

        #[arg(long, default_value_t = 1)]
        max_latency: u64,

        #[arg(long, default_value_t = 1)]
        residual_ii: u64,

        #[arg(long, default_value_t = 1)]
        residual_latency: u64,

        #[arg(long, default_value_t = 1)]
        vector_prod_ii: u64,

        #[arg(long, default_value_t = 1)]
        vector_prod_latency: u64,
    },
}

#[derive(Debug, Args)]
struct CommonTimings {
    /// Matmul initiation interval
    #[arg(long, default_value_t = 1)]
    matmul_ii: u64,

    /// Matmul latency
    #[arg(long, default_value_t = 1)]
    matmul_latency: u64,

    #[arg(long, default_value_t = 1)]
    div_ii: u64,

    #[arg(long, default_value_t = 1)]
    div_latency: u64,

    #[arg(long, default_value_t = 0)]
    reset_time: u64,
}

fn main() {
    let args = CommandLineInterface::parse();
    println!("{:?}", args);

    let gen_start = std::time::Instant::now();

    // Construct the QKT multiplies
    let q_matrices = (0..args.batch)
        .map(|_| ArcArray::from_shape_simple_fn([args.length, args.dim], fastrand::f32))
        .collect::<Vec<_>>();
    let k_matrices = (0..args.batch)
        .map(|_| ArcArray::from_shape_simple_fn([args.length, args.dim], fastrand::f32))
        .collect::<Vec<_>>();

    let v_matrices = (0..args.batch)
        .map(|_| ArcArray::from_shape_simple_fn([args.length, args.dim], fastrand::f32))
        .collect::<Vec<_>>();

    println!("Took {:?} to generate random values", gen_start.elapsed());

    let short_depth = match args.mode {
        Implementation::Naive { short_depth, .. } => short_depth,
        Implementation::Agnostic { channel_depth, .. } => channel_depth,
    };

    let mut builder = ProgramBuilder::default();

    let (qkt_receiver, v_receiver) = {
        let (qkt_sender, qkt_receiver) = builder.bounded(short_depth);

        let (a_snd, a_recv) = builder.bounded(short_depth);
        let (b_snd, b_recv) = builder.bounded(short_depth);

        builder.add_child(GeneratorContext::new(
            || q_matrices.iter().flat_map(|mat| mat.into_iter()).copied(),
            a_snd,
        ));
        builder.add_child(GeneratorContext::new(
            || {
                k_matrices.iter().flat_map(|mat_b| {
                    (0..args.length)
                        .flat_map(move |_| mat_b.iter().copied().collect::<Vec<_>>().into_iter())
                })
            },
            b_snd,
        ));

        builder.add_child(Matmul::new(
            MatmulTiming {
                dot_latency: args.common.matmul_latency,
                dot_ii: args.common.matmul_ii,
                reset_time: args.common.reset_time,
            },
            crate::templates::MatmulBehavior::Buffered,
            ShapeInfo {
                m: args.length,
                n: args.length,
                k: args.dim,
            },
            a_recv,
            b_recv,
            qkt_sender,
            |a, b, c| a * b + c,
        ));

        let (v_snd, v_recv) = builder.bounded(short_depth);
        builder.add_child(GeneratorContext::new(
            || {
                v_matrices.iter().flat_map(|v| {
                    (0..args.length).flat_map(move |_| match args.mode {
                        Implementation::Naive { .. } => {
                            v.t().iter().copied().collect::<Vec<_>>().into_iter()
                        }
                        Implementation::Agnostic { .. } => {
                            v.iter().copied().collect::<Vec<_>>().into_iter()
                        }
                    })
                })
            },
            v_snd,
        ));

        (qkt_receiver, v_recv)
    };

    let config = AttentionConfig {
        vocab_dim: args.dim,
        seq_len: args.length,
    };

    let output = match args.mode {
        Implementation::Naive {
            short_depth,
            long_depth,
            exp_ii,
            exp_latency,
            sum_ii,
            sum_latency,
        } => {
            if long_depth < args.length {
                println!(
                    "Warning: Long Depth is shorter than the sequence length, this will deadlock."
                );
            }
            apps::naive::naive(
                &mut builder,
                qkt_receiver,
                v_receiver,
                config,
                apps::naive::NaiveConfig {
                    long_chan_size: long_depth,
                    short_chan_depth: short_depth,
                    exp_timings: MapTimings {
                        initiation_interval: exp_ii,
                        latency: exp_latency,
                    },
                    div_timings: MapTimings {
                        initiation_interval: args.common.div_ii,
                        latency: args.common.div_latency,
                    },
                    sum_timings: ReduceTimings {
                        initiation_interval: sum_ii,
                        latency: sum_latency,
                        reset_time: args.common.reset_time,
                    },
                    matmul_timings: MatmulTiming {
                        dot_latency: args.common.matmul_latency,
                        dot_ii: args.common.matmul_ii,
                        reset_time: args.common.reset_time,
                    },
                },
            )
        }
        Implementation::Agnostic {
            channel_depth,
            max_ii,
            max_latency,
            residual_ii,
            residual_latency,
            vector_prod_ii,
            vector_prod_latency,
        } => apps::agnostic::agnostic_attention(
            &mut builder,
            qkt_receiver,
            v_receiver,
            config,
            AgnosticConfig {
                chan_depth: channel_depth,
                max_config: ScanTimings {
                    initiation_interval: max_ii,
                    latency: max_latency,
                    reset_time: args.common.reset_time,
                },
                residual_config: ReduceTimings {
                    initiation_interval: residual_ii,
                    latency: residual_latency,
                    reset_time: args.common.reset_time,
                },
                prod_config: ReduceTimings {
                    initiation_interval: vector_prod_ii,
                    latency: vector_prod_latency,
                    reset_time: args.common.reset_time,
                },
                scale_config: FlatmapTimings {
                    initiation_interval: args.common.div_ii,
                    latency: args.common.div_latency,
                },
            },
        ),
    };
    if args.validate {
        builder.add_child(ApproxCheckerContext::new(
            || {
                let validation_matrices =
                    izip!(q_matrices.iter(), k_matrices.iter(), v_matrices.iter());
                let golds = validation_matrices
                    .map(|(q, k, v)| compute_attention(q.view(), k.view(), v.view()));
                golds.flat_map(|gold| gold.into_iter())
            },
            output,
            |a, b| (a - b).abs() < 0.01,
        ));
    } else {
        builder.add_child(ConsumerContext::new(output));
    }

    let run_opts = match args.workers {
        Some(workers) => RunOptionsBuilder::default()
            .mode(RunMode::Constrained(workers))
            .build()
            .unwrap(),
        None => Default::default(),
    };

    let executed = builder
        .initialize(Default::default())
        .expect("Failed to initialize and validate graph")
        .run(run_opts);
    println!("Elapsed Cycles: {}", executed.elapsed_cycles().unwrap());
}
