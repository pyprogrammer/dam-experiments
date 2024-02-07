use dam::context_tools::*;

use super::BroadcastSender;

#[context_macro]
pub struct Repeat<InT: DAMType> {
    input: Receiver<InT>,
    output: BroadcastSender<InT>,
    repeats: usize,
}

impl<InT: DAMType> Repeat<InT>
where
    Self: Context,
{
    pub fn new(input: Receiver<InT>, output: BroadcastSender<InT>, repeats: usize) -> Self {
        let s = Self {
            input,
            output,
            repeats,
            context_info: Default::default(),
        };
        s.input.attach_receiver(&s);
        s.output.attach_sender(&s);
        s
    }
}

impl<InT: DAMType> Context for Repeat<InT> {
    fn run(&mut self) {
        loop {
            match self.input.dequeue(&self.time) {
                Ok(ChannelElement { time: _, data }) => {
                    for _ in 0..self.repeats {
                        self.output
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: self.time.tick() + 1,
                                    data: data.clone(),
                                },
                            )
                            .unwrap();
                    }
                }
                Err(_) => return,
            }
        }
    }
}
