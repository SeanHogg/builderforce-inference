use crate::error::CoreError;
use crate::DType;

#[derive(Clone, Debug)]
pub enum Storage {
    F32(Vec<f32>),
    Quant { ggml_id: u32, bytes: Vec<u8> },
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub storage: Storage,
}

impl Tensor {
    pub fn f32(shape: Vec<usize>, data: Vec<f32>) -> Self {
        debug_assert_eq!(data.len(), n_elems_of(&shape));
        Self {
            shape,
            dtype: DType::F32,
            storage: Storage::F32(data),
        }
    }

    pub fn n_elems(&self) -> usize {
        n_elems_of(&self.shape)
    }

    pub fn as_f32(&self) -> anyhow::Result<&[f32]> {
        match &self.storage {
            Storage::F32(v) => Ok(v),
            Storage::Quant { .. } => Err(CoreError::ExpectedF32(self.dtype).into()),
        }
    }

    pub fn as_f32_mut(&mut self) -> anyhow::Result<&mut [f32]> {
        match &mut self.storage {
            Storage::F32(v) => Ok(v),
            Storage::Quant { .. } => Err(CoreError::ExpectedF32(self.dtype).into()),
        }
    }
}

pub(crate) fn n_elems_of(shape: &[usize]) -> usize {
    shape.iter().product()
}
