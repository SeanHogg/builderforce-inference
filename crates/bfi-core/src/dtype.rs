use bfi_gguf::{
    TYPE_BF16, TYPE_F16, TYPE_F32, TYPE_PQ2_0, TYPE_PTQ1_0, TYPE_Q1_0, TYPE_Q4_0, TYPE_Q4_K,
    TYPE_Q5_K, TYPE_Q6_K, TYPE_Q8_0,
};

// Variant names mirror GGML type names (Q4_0, PQ2_0, …).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
    PQ2_0,
    PTQ1_0,
    /// PrismML 1-bit (Bonsai), GGML id 41.
    Q1_0,
}

impl DType {
    pub fn from_ggml(id: u32) -> Option<Self> {
        Some(match id {
            TYPE_F32 => DType::F32,
            TYPE_F16 => DType::F16,
            TYPE_BF16 => DType::BF16,
            TYPE_Q4_0 => DType::Q4_0,
            TYPE_Q8_0 => DType::Q8_0,
            TYPE_Q4_K => DType::Q4_K,
            TYPE_Q5_K => DType::Q5_K,
            TYPE_Q6_K => DType::Q6_K,
            TYPE_PQ2_0 => DType::PQ2_0,
            TYPE_PTQ1_0 => DType::PTQ1_0,
            TYPE_Q1_0 => DType::Q1_0,
            _ => return None,
        })
    }

    pub fn ggml_id(self) -> u32 {
        match self {
            DType::F32 => TYPE_F32,
            DType::F16 => TYPE_F16,
            DType::BF16 => TYPE_BF16,
            DType::Q4_0 => TYPE_Q4_0,
            DType::Q8_0 => TYPE_Q8_0,
            DType::Q4_K => TYPE_Q4_K,
            DType::Q5_K => TYPE_Q5_K,
            DType::Q6_K => TYPE_Q6_K,
            DType::PQ2_0 => TYPE_PQ2_0,
            DType::PTQ1_0 => TYPE_PTQ1_0,
            DType::Q1_0 => TYPE_Q1_0,
        }
    }
}
