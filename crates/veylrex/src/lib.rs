//! In-process orchestration for optional model-runner crates.
//!
//! The target API is defined in the Veylrex specification. This crate is currently a scaffold.

pub mod capability;
pub mod provider;

pub use capability::{
    ComputeDevice, DeviceClass, ProviderCapability, ProviderId, ProviderRequest,
    ProviderSelectionError, ProviderStatus, SystemCapabilities, host_capabilities,
};
pub use provider::{select_provider, select_provider_for_engine};

/// Return host facts and the provider rows for enabled Veylrex engines.
pub fn system_capabilities() -> SystemCapabilities {
    #[cfg(feature = "mnn")]
    {
        let mut capabilities = host_capabilities();
        capabilities.providers.extend(mnn::capabilities());
        #[cfg(feature = "onnx")]
        capabilities.providers.extend(onnx_capabilities());
        #[cfg(feature = "llama-cpp")]
        capabilities.providers.extend(llama_capabilities());
        capabilities
    }
    #[cfg(not(feature = "mnn"))]
    {
        let mut capabilities = host_capabilities();
        #[cfg(feature = "onnx")]
        capabilities.providers.extend(onnx_capabilities());
        #[cfg(feature = "llama-cpp")]
        capabilities.providers.extend(llama_capabilities());
        capabilities
    }
}

#[cfg(feature = "llama-cpp")]
fn llama_capabilities() -> Vec<ProviderCapability> {
    llama_cpp_oxide::capabilities()
        .into_iter()
        .map(|capability| {
            let provider = match capability.provider {
                llama_cpp_oxide::Provider::Cpu => ProviderId::Cpu,
                llama_cpp_oxide::Provider::Cuda => ProviderId::Cuda,
                llama_cpp_oxide::Provider::Vulkan => ProviderId::Vulkan,
                llama_cpp_oxide::Provider::Metal => ProviderId::Metal,
            };
            let implementation = if capability.compiled {
                ProviderStatus::Available
            } else {
                ProviderStatus::NotCompiled
            };
            let runtime = match capability.available {
                Some(true) if capability.reason.is_none() => ProviderStatus::Available,
                Some(true) => ProviderStatus::Unknown {
                    reason: capability
                        .reason
                        .unwrap_or_else(|| "runtime probe was incomplete".into()),
                },
                Some(false) => ProviderStatus::Unavailable {
                    reason: capability
                        .reason
                        .unwrap_or_else(|| "llama.cpp backend is unavailable".into()),
                },
                None => ProviderStatus::Unknown {
                    reason: capability
                        .reason
                        .unwrap_or_else(|| "llama.cpp runtime probe failed".into()),
                },
            };
            ProviderCapability {
                engine: "llama-cpp-oxide".into(),
                provider,
                implementation,
                runtime,
                details: Some(
                    "GPU offload or model execution does not establish per-operator placement"
                        .into(),
                ),
                device_ids: Vec::new(),
            }
        })
        .collect()
}

#[cfg(feature = "onnx")]
fn onnx_capabilities() -> Vec<ProviderCapability> {
    onnx_oxide::capabilities()
        .into_iter()
        .map(|capability| {
            let provider = match capability.provider {
                onnx_oxide::Provider::Cpu => ProviderId::Cpu,
                onnx_oxide::Provider::Cuda => ProviderId::Cuda,
                onnx_oxide::Provider::CoreMl => ProviderId::CoreMl,
            };
            let implementation = if capability.implemented {
                ProviderStatus::Available
            } else {
                ProviderStatus::NotCompiled
            };
            let runtime = match capability.probe {
                onnx_oxide::ProviderProbe::RuntimeNotInitialized => ProviderStatus::Unknown {
                    reason: "ONNX Runtime shared library has not been initialized".into(),
                },
                onnx_oxide::ProviderProbe::RuntimeRegistered => ProviderStatus::Unknown {
                    reason: "ONNX Runtime includes this provider, but no session has confirmed system usability".into(),
                },
                onnx_oxide::ProviderProbe::RuntimeNotRegistered => ProviderStatus::Unavailable {
                    reason: "loaded ONNX Runtime does not register this Execution Provider".into(),
                },
                onnx_oxide::ProviderProbe::SessionCreated => ProviderStatus::Available,
                onnx_oxide::ProviderProbe::Error(reason) => ProviderStatus::Unknown { reason },
            };
            ProviderCapability {
                engine: "onnx-oxide".into(),
                provider,
                implementation,
                runtime,
                details: Some("session registration does not establish placement for every model operator".into()),
                device_ids: Vec::new(),
            }
        })
        .collect()
}

#[cfg(feature = "mnn")]
pub mod mnn;
#[cfg(feature = "llama-cpp")]
pub use llama_cpp_oxide as llama_cpp;
#[cfg(feature = "onnx")]
pub use onnx_oxide as onnx;
pub mod vision;
pub use vision::{VisionInput, VisionLayout, VisionOutput, VisionTensor};
