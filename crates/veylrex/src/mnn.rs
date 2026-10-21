//! MNN implementation of the preprocessed vision tensor contract.
use crate::{
    capability::*,
    select_provider_for_engine,
    vision::{VisionInput, VisionLayout, VisionOutput, VisionTensor},
};

/// MNN-specific session controls retained by the adapter.
#[derive(Debug, Clone, Copy)]
pub struct SessionOptions {
    /// MNN inference worker count.
    pub threads: u32,
    /// Memory use preference.
    pub memory: mnn_oxide::MemoryMode,
    /// Scheduling power preference.
    pub power: mnn_oxide::PowerMode,
    /// Arithmetic precision preference.
    pub precision: mnn_oxide::PrecisionMode,
}
impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            threads: 4,
            memory: mnn_oxide::MemoryMode::Normal,
            power: mnn_oxide::PowerMode::Normal,
            precision: mnn_oxide::PrecisionMode::Normal,
        }
    }
}

/// Report MNN provider build/runtime state without treating adapter features as native proof.
pub fn capabilities() -> Vec<ProviderCapability> {
    mnn_oxide::BackendType::PROVIDERS
        .iter()
        .map(|backend| {
            let cap = backend.capability();
            let provider = match backend {
                mnn_oxide::BackendType::CPU => ProviderId::Cpu,
                mnn_oxide::BackendType::Cuda => ProviderId::Cuda,
                mnn_oxide::BackendType::Vulkan => ProviderId::Vulkan,
                mnn_oxide::BackendType::Metal => ProviderId::Metal,
                mnn_oxide::BackendType::OpenCL => ProviderId::MnnOpenCl,
                mnn_oxide::BackendType::Auto => unreachable!(),
            };
            let implementation = if !cap.adapter_enabled {
                ProviderStatus::NotCompiled
            } else {
                match backend.probe_native_compiled() {
                    Ok(true) => ProviderStatus::Available,
                    Ok(false) => ProviderStatus::NotCompiled,
                    Err(error) => ProviderStatus::Unknown {
                        reason: error.to_string(),
                    },
                }
            };
            let runtime = if !cap.adapter_enabled {
                ProviderStatus::NotCompiled
            } else if !matches!(&implementation, ProviderStatus::Available) {
                implementation.clone()
            } else {
                match backend.probe_native_runtime() {
                    Ok(actual) if actual == *backend => ProviderStatus::Available,
                    Ok(actual) => ProviderStatus::Unavailable {
                        reason: format!(
                            "MNN assigned {} instead of requested {}",
                            actual.name(),
                            backend.name()
                        ),
                    },
                    Err(error) => ProviderStatus::Unknown {
                        reason: error.to_string(),
                    },
                }
            };
            ProviderCapability {
                engine: "mnn-oxide".into(),
                provider,
                implementation,
                runtime,
                details: cap.reason.map(str::to_owned),
                device_ids: Vec::new(),
            }
        })
        .collect()
}

/// Run one preprocessed f32 image tensor with MNN.
pub fn run(input: VisionInput) -> Result<VisionOutput, Box<dyn std::error::Error + Send + Sync>> {
    run_with_options(input, SessionOptions::default())
}

/// Run one preprocessed f32 image tensor with explicit MNN controls.
pub fn run_with_options(
    input: VisionInput,
    options: SessionOptions,
) -> Result<VisionOutput, Box<dyn std::error::Error + Send + Sync>> {
    let n = checked_count(&input.shape)?;
    if input.data.len() != n {
        return Err(format!(
            "input data has {} values but shape requires {n}",
            input.data.len()
        )
        .into());
    }
    if input.shape.len() > 9 {
        return Err("MNN tensor rank cannot exceed 9".into());
    }
    let providers = capabilities();
    let caps = SystemCapabilities {
        providers,
        ..SystemCapabilities::default()
    };
    let selected = select_provider_for_engine(&caps, "mnn-oxide", &input.provider)?;
    let requested = match &input.provider {
        crate::ProviderRequest::Explicit(id) => id.clone(),
        crate::ProviderRequest::Unspecified => selected.clone(),
    };
    let backend = match selected {
        ProviderId::Cpu => mnn_oxide::BackendType::CPU,
        ProviderId::Cuda => mnn_oxide::BackendType::Cuda,
        ProviderId::Vulkan => mnn_oxide::BackendType::Vulkan,
        ProviderId::Metal => mnn_oxide::BackendType::Metal,
        ProviderId::MnnOpenCl => mnn_oxide::BackendType::OpenCL,
        _ => return Err(format!("provider {selected:?} is not an MNN provider").into()),
    };
    let interpreter = mnn_oxide::Interpreter::from_file(&input.model)?;
    let config = mnn_oxide::ScheduleConfig::new()
        .backend(backend)
        .num_threads(options.threads)
        .memory_mode(options.memory)
        .power_mode(options.power)
        .precision_mode(options.precision);
    let mut session = interpreter.create_session(config)?;
    let input_tensor = session.get_input(input.input_name.as_deref())?;
    let info = input_tensor.info()?;
    if info.shape != input.shape {
        return Err(format!(
            "model input shape {:?} differs from provided {:?}",
            info.shape, input.shape
        )
        .into());
    }
    if info.dtype != mnn_oxide::DataType::Float32 {
        return Err(format!(
            "MNN vision adapter currently accepts only f32 model inputs; found {}",
            info.dtype.name()
        )
        .into());
    }
    // MNN's tensor format can describe backend storage packing (for example
    // NC4HW4) rather than the model's logical axis order. The caller supplies
    // logical layout explicitly; MNN handles transfer into its internal form.
    let _logical_layout = mnn_layout(input.layout);
    input_tensor.write(&input.data)?;
    session.run()?;
    let output_names = if input.output_names.is_empty() {
        vec![None]
    } else {
        input
            .output_names
            .iter()
            .map(|name| Some(name.as_str()))
            .collect()
    };
    let mut outputs = Vec::with_capacity(output_names.len());
    for name in output_names {
        let output = session.get_output(name)?;
        let out_info = output.info()?;
        if out_info.dtype != mnn_oxide::DataType::Float32 {
            return Err(format!(
                "MNN vision adapter currently returns only f32 outputs; found {}",
                out_info.dtype.name()
            )
            .into());
        }
        outputs.push(VisionTensor {
            data: output.read::<f32>()?,
            shape: out_info.shape,
            dtype: out_info.dtype.name().into(),
        });
    }
    let native = session.backend()?;
    let native_provider = match native {
        mnn_oxide::BackendType::CPU => ProviderId::Cpu,
        mnn_oxide::BackendType::Cuda => ProviderId::Cuda,
        mnn_oxide::BackendType::Vulkan => ProviderId::Vulkan,
        mnn_oxide::BackendType::Metal => ProviderId::Metal,
        mnn_oxide::BackendType::OpenCL => ProviderId::MnnOpenCl,
        mnn_oxide::BackendType::Auto => ProviderId::Engine {
            engine: "mnn-oxide".into(),
            name: "Auto".into(),
        },
    };
    Ok(VisionOutput {
        outputs,
        requested_provider: requested,
        native_provider,
    })
}

fn checked_count(shape: &[i32]) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    if shape.is_empty() || shape.iter().any(|d| *d <= 0) {
        return Err("tensor shape must have positive dimensions".into());
    }
    shape.iter().try_fold(1usize, |n, d| {
        n.checked_mul(*d as usize)
            .ok_or_else(|| "tensor size overflow".into())
    })
}

/// Convert Veylrex layout to the MNN format for downstream metadata.
pub fn mnn_layout(layout: VisionLayout) -> mnn_oxide::DataFormat {
    match layout {
        VisionLayout::Nhwc => mnn_oxide::DataFormat::Nhwc,
        VisionLayout::Nchw => mnn_oxide::DataFormat::Nchw,
    }
}
