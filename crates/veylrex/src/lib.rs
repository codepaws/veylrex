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
        capabilities
    }
    #[cfg(not(feature = "mnn"))]
    {
        host_capabilities()
    }
}

#[cfg(feature = "mnn")]
pub mod mnn;
pub mod vision;
pub use vision::{VisionInput, VisionLayout, VisionOutput, VisionTensor};
