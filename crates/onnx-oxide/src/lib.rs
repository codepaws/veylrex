//! Explicitly loaded ONNX Runtime runner.
//!
//! Runtime binaries are never downloaded by this crate. Applications provide
//! the ONNX Runtime shared-library path before loading a model. CUDA and CoreML
//! are exposed only when the matching `ort` feature is enabled.

use std::{
    error::Error as StdError,
    path::{Path, PathBuf},
    sync::Mutex,
};

/// ONNX Runtime execution provider request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// ONNX Runtime CPU execution provider.
    Cpu,
    /// ONNX Runtime CUDA execution provider.
    Cuda,
    /// ONNX Runtime CoreML execution provider.
    CoreMl,
}

/// ONNX Runtime provider probe result. A successful session probe proves the
/// requested provider accepted a session, not that every model node runs there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderProbe {
    /// The runtime library has not been initialized by this process.
    RuntimeNotInitialized,
    /// The runtime reports this provider compiled in, but no session has
    /// confirmed it can be registered on this host.
    RuntimeRegistered,
    /// The loaded runtime does not expose this provider.
    RuntimeNotRegistered,
    /// A model session accepted this provider on this host.
    SessionCreated,
    /// The runtime provider query failed.
    Error(String),
}

/// Build and runtime state for one ONNX Runtime provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderCapability {
    /// Provider name in ONNX Runtime terminology.
    pub provider: Provider,
    /// Whether this `onnx-oxide` build has the adapter/provider feature.
    pub implemented: bool,
    /// Native runtime evidence.
    pub probe: ProviderProbe,
}

/// Owned f32 model input with explicit shape.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInput {
    /// ONNX input name.
    pub name: String,
    /// Concrete tensor dimensions.
    pub shape: Vec<i64>,
    /// Contiguous row-major tensor values.
    pub data: Vec<f32>,
}

impl TensorInput {
    /// Validate and construct a contiguous f32 input tensor.
    pub fn new(
        name: impl Into<String>,
        shape: Vec<i64>,
        data: Vec<f32>,
    ) -> Result<Self, InputError> {
        let name = name.into();
        if name.is_empty() {
            return Err(InputError("input name must not be empty".into()));
        }
        if shape.is_empty() || shape.iter().any(|dimension| *dimension <= 0) {
            return Err(InputError("input dimensions must be positive".into()));
        }
        let count = shape
            .iter()
            .try_fold(1usize, |count, dimension| {
                count.checked_mul(*dimension as usize)
            })
            .ok_or_else(|| InputError("input element count overflows usize".into()))?;
        if data.len() != count {
            return Err(InputError(format!(
                "input data has {} values but shape requires {count}",
                data.len()
            )));
        }
        Ok(Self { name, shape, data })
    }
}

/// Invalid input tensor contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputError(String);

impl std::fmt::Display for InputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl StdError for InputError {}

/// Owned f32 tensor returned from an ONNX model.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorOutput {
    /// ONNX output name.
    pub name: String,
    /// Concrete tensor dimensions.
    pub shape: Vec<i64>,
    /// Contiguous row-major tensor values.
    pub data: Vec<f32>,
}

/// ONNX Runtime model session. `configured_provider` records the requested
/// execution provider; ONNX Runtime may partition operators across providers.
pub struct Runner {
    session: ort::session::Session,
    configured_provider: Provider,
}

static RUNTIME_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
static SESSION_VERIFIED: Mutex<Vec<Provider>> = Mutex::new(Vec::new());

/// Report provider build state and the strongest runtime evidence collected so
/// far by this process.
pub fn capabilities() -> Vec<ProviderCapability> {
    [Provider::Cpu, Provider::Cuda, Provider::CoreMl]
        .into_iter()
        .map(|provider| {
            let implemented = match provider {
                Provider::Cpu => true,
                Provider::Cuda => cfg!(feature = "cuda"),
                Provider::CoreMl => cfg!(feature = "coreml"),
            };
            let probe = if !implemented {
                ProviderProbe::RuntimeNotRegistered
            } else if SESSION_VERIFIED
                .lock()
                .map(|verified| verified.contains(&provider))
                .unwrap_or(false)
            {
                ProviderProbe::SessionCreated
            } else if RUNTIME_PATH
                .lock()
                .map(|path| path.is_none())
                .unwrap_or(true)
            {
                ProviderProbe::RuntimeNotInitialized
            } else {
                match provider_is_registered(provider) {
                    Ok(true) => ProviderProbe::RuntimeRegistered,
                    Ok(false) => ProviderProbe::RuntimeNotRegistered,
                    Err(error) => ProviderProbe::Error(error.to_string()),
                }
            };
            ProviderCapability {
                provider,
                implemented,
                probe,
            }
        })
        .collect()
}

impl Runner {
    /// Load ONNX Runtime from a caller-supplied shared library and create a
    /// model session using the requested provider. Runtime initialization is
    /// process-global; subsequent calls must use the same library path.
    pub fn load(
        runtime_library: impl AsRef<Path>,
        model_path: impl AsRef<Path>,
        provider: Provider,
    ) -> Result<Self, Box<dyn StdError>> {
        let dispatch = provider_dispatch(provider)?;
        initialize_runtime(runtime_library.as_ref())?;
        let session = ort::session::Session::builder()?
            .with_execution_providers([dispatch])?
            .commit_from_file(model_path)?;
        if let Ok(mut verified) = SESSION_VERIFIED.lock() {
            if !verified.contains(&provider) {
                verified.push(provider);
            }
        }
        Ok(Self {
            session,
            configured_provider: provider,
        })
    }

    /// Return the requested provider configuration. This does not claim that
    /// every node was assigned to that provider.
    pub fn configured_provider(&self) -> Provider {
        self.configured_provider
    }

    /// Execute one named input and return all f32 outputs as owned buffers.
    pub fn run(&mut self, input: TensorInput) -> Result<Vec<TensorOutput>, Box<dyn StdError>> {
        let tensor = ort::value::Tensor::from_array((input.shape, input.data))?;
        let outputs = self
            .session
            .run(ort::inputs![input.name.as_str() => tensor])?;
        outputs
            .into_iter()
            .map(|(name, value)| {
                let (shape, values) = value.try_extract_tensor::<f32>()?;
                Ok(TensorOutput {
                    name: name.to_owned(),
                    shape: shape.iter().copied().collect(),
                    data: values.to_vec(),
                })
            })
            .collect()
    }
}

fn provider_is_registered(provider: Provider) -> ort::Result<bool> {
    use ort::ep::ExecutionProvider;
    match provider {
        Provider::Cpu => ort::ep::CPU::default().is_available(),
        Provider::Cuda => {
            #[cfg(feature = "cuda")]
            {
                ort::ep::CUDA::default().is_available()
            }
            #[cfg(not(feature = "cuda"))]
            {
                Ok(false)
            }
        }
        Provider::CoreMl => {
            #[cfg(feature = "coreml")]
            {
                ort::ep::CoreML::default().is_available()
            }
            #[cfg(not(feature = "coreml"))]
            {
                Ok(false)
            }
        }
    }
}

fn initialize_runtime(path: &Path) -> Result<(), Box<dyn StdError>> {
    let path = path.canonicalize()?;
    let mut initialized = RUNTIME_PATH
        .lock()
        .map_err(|_| "ONNX Runtime initialization lock is poisoned")?;
    if let Some(current) = initialized.as_ref() {
        if current != &path {
            return Err(format!(
                "ONNX Runtime is already initialized from {}; cannot switch to {}",
                current.display(),
                path.display()
            )
            .into());
        }
        return Ok(());
    }
    ort::init_from(&path)?.commit();
    *initialized = Some(path);
    Ok(())
}

fn provider_dispatch(
    provider: Provider,
) -> Result<ort::ep::ExecutionProviderDispatch, Box<dyn StdError>> {
    match provider {
        Provider::Cpu => Ok(ort::ep::CPU::default().build().error_on_failure()),
        Provider::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(ort::ep::CUDA::default().build().error_on_failure())
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("ONNX Runtime CUDA provider was not compiled in".into())
            }
        }
        Provider::CoreMl => {
            #[cfg(feature = "coreml")]
            {
                Ok(ort::ep::CoreML::default().build().error_on_failure())
            }
            #[cfg(not(feature = "coreml"))]
            {
                Err("ONNX Runtime CoreML provider was not compiled in".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_and_nonpositive_shapes() {
        assert!(TensorInput::new("x", vec![], vec![]).is_err());
        assert!(TensorInput::new("x", vec![1, 0], vec![]).is_err());
    }

    #[test]
    fn explicit_uncompiled_provider_is_rejected_before_runtime_loading() {
        if !cfg!(feature = "cuda") {
            let error = provider_dispatch(Provider::Cuda).unwrap_err().to_string();
            assert!(error.contains("not compiled"));
        }
    }
}
