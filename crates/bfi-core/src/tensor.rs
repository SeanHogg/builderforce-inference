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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_tensor_exposes_shape_dtype_and_elems() {
        let t = Tensor::f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.n_elems(), 6);
        assert_eq!(t.as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn quant_storage_is_not_f32() {
        let t = Tensor {
            shape: vec![128],
            dtype: DType::PTQ1_0,
            storage: Storage::Quant {
                ggml_id: DType::PTQ1_0.ggml_id(),
                bytes: vec![0u8; 28],
            },
        };
        assert_eq!(t.dtype, DType::PTQ1_0);
        assert_eq!(t.n_elems(), 128);
        assert!(t.as_f32().is_err());
        let mut t = t;
        assert!(t.as_f32_mut().is_err());
    }

    #[test]
    fn clone_copies_storage() {
        let t = Tensor::f32(vec![2], vec![3.0, 4.0]);
        let mut c = t.clone();
        c.as_f32_mut().unwrap()[0] = 9.0;
        assert_eq!(t.as_f32().unwrap(), &[3.0, 4.0]);
        assert_eq!(c.as_f32().unwrap(), &[9.0, 4.0]);
    }
}
