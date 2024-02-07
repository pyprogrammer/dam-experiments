use std::ops::Deref;

use dam::{
    channel::EnqueueError,
    context_tools::{ChannelElement, Context, Sender},
    structures::TimeManager,
    types::DAMType,
};
use ndarray::{linalg::Dot, ArcArray, Dimension};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor<A, D: Dimension>(pub ArcArray<A, D>);

impl<A, D: Dimension> Deref for Tensor<A, D> {
    type Target = ArcArray<A, D>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<A, D: Dimension, T> From<T> for Tensor<A, D>
where
    T: Into<ArcArray<A, D>>,
{
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

impl<A, D: Dimension> DAMType for Tensor<A, D>
where
    A: DAMType,
{
    fn dam_size(&self) -> usize {
        todo!()
    }
}

impl<A, D: Dimension> Default for Tensor<A, D>
where
    A: DAMType,
{
    fn default() -> Self {
        todo!()
    }
}

impl<A, B, D, DB> Dot<Tensor<B, DB>> for Tensor<A, D>
where
    ArcArray<A, D>: Dot<ArcArray<B, DB>>,
    D: Dimension,
    DB: Dimension,
{
    type Output = <ArcArray<A, D> as Dot<ArcArray<B, DB>>>::Output;

    fn dot(&self, rhs: &Tensor<B, DB>) -> Self::Output {
        self.0.dot(&rhs.0)
    }
}

pub struct BroadcastSender<T: DAMType> {
    pub targets: Vec<Sender<T>>,
}

impl<T: DAMType> BroadcastSender<T> {
    pub fn attach_sender(&self, ctx: &dyn Context) {
        self.targets.iter().for_each(|chn| chn.attach_sender(ctx));
    }

    pub fn enqueue(&self, time: &TimeManager, data: ChannelElement<T>) -> Result<(), EnqueueError> {
        if self
            .targets
            .iter()
            .map(|chn| chn.wait_until_available(time))
            .any(|wait_result| wait_result.is_err())
        {
            return Err(EnqueueError::Closed);
        }
        self.targets.iter().for_each(|chn| {
            let _ = chn.enqueue(time, data.clone());
        });
        Ok(())
    }
}
