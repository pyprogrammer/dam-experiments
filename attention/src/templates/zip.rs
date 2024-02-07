use dam::context_tools::*;

use super::BroadcastSender;

#[derive(Debug, Default, Clone, Copy)]
pub struct Pair<A, B>(pub A, pub B);

impl<A: DAMType, B: DAMType> DAMType for Pair<A, B> {
    fn dam_size(&self) -> usize {
        self.0.dam_size() + self.1.dam_size()
    }
}

#[context_macro]
pub struct Zip<LeftT: DAMType, RightT: DAMType> {
    left: Receiver<LeftT>,
    right: Receiver<RightT>,
    output: BroadcastSender<Pair<LeftT, RightT>>,
}

impl<LeftT: DAMType, RightT: DAMType> Zip<LeftT, RightT>
where
    Self: Context,
{
    pub fn new(
        left: Receiver<LeftT>,
        right: Receiver<RightT>,
        output: BroadcastSender<Pair<LeftT, RightT>>,
    ) -> Self {
        let s = Self {
            left,
            right,
            output,
            context_info: Default::default(),
        };
        s.left.attach_receiver(&s);
        s.right.attach_receiver(&s);
        s.output.attach_sender(&s);
        s
    }
}

impl<LeftT: DAMType, RightT: DAMType> Context for Zip<LeftT, RightT> {
    fn run(&mut self) {
        loop {
            let _ = self.left.peek_next(&self.time);
            let _ = self.right.peek_next(&self.time);
            match (
                self.left.dequeue(&self.time),
                self.right.dequeue(&self.time),
            ) {
                (Ok(l), Ok(r)) => self
                    .output
                    .enqueue(
                        &self.time,
                        ChannelElement {
                            time: self.time.tick() + 1,
                            data: Pair(l.data, r.data),
                        },
                    )
                    .unwrap(),
                (Err(_), Err(_)) => return,
                (l, r) => panic!(
                    "Mismatched left and right for zip {:?}: L({:?}), R({:?})",
                    self.id,
                    l.is_ok(),
                    r.is_ok()
                ),
            }
        }
    }
}
