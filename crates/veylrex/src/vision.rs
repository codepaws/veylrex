//! Preprocessed tensor contract for vision inference.
use crate::{ProviderId, ProviderRequest};

/// Tensor storage layout supplied by the model contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionLayout {
    /// Batch, height, width, channels.
    Nhwc,
    /// Batch, channels, height, width.
    Nchw,
}

/// Input image tensor already decoded, resized, normalized, and laid out by the caller.
#[derive(Debug, Clone)]
pub struct VisionInput {
    /// MNN or engine model path.
    pub model: std::path::PathBuf,
    /// Tensor values in row-major order.
    pub data: Vec<f32>,
    /// Explicit tensor dimensions.
    pub shape: Vec<i32>,
    /// Tensor layout.
    pub layout: VisionLayout,
    /// Input name, or `None` for the first input.
    pub input_name: Option<String>,
    /// Requested output names; empty selects the engine's first output.
    pub output_names: Vec<String>,
    /// Requested compute provider.
    pub provider: ProviderRequest,
}
/// One owned raw output tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionTensor {
    /// Tensor values.
    pub data: Vec<f32>,
    /// Tensor dimensions.
    pub shape: Vec<i32>,
    /// Data type name.
    pub dtype: String,
}
/// Owned output tensors and provider assignment for one inference.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionOutput {
    /// Output tensors returned by the engine.
    pub outputs: Vec<VisionTensor>,
    /// Provider requested by caller or selected by policy.
    pub requested_provider: ProviderId,
    /// Provider reported by the native session.
    pub native_provider: ProviderId,
}
