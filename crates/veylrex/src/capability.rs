//! Shared, conservative runtime and device capability records.
use std::fmt;

/// Provider identifier. Engine-specific providers retain their native name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProviderId {
    /// CPU execution.
    Cpu,
    /// NVIDIA CUDA.
    Cuda,
    /// Vulkan compute.
    Vulkan,
    /// Apple Metal.
    Metal,
    /// Apple Core ML.
    CoreMl,
    /// MNN OpenCL.
    MnnOpenCl,
    /// Provider unique to one engine.
    Engine {
        /// Engine identifier.
        engine: String,
        /// Native provider identifier.
        name: String,
    },
}

/// Build and runtime state for a provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderStatus {
    /// Runner has no implementation for this provider.
    NotImplemented,
    /// Implementation exists but was excluded from the build.
    NotCompiled,
    /// Runtime probe did not establish availability.
    Unknown {
        /// Explanation for the unknown result.
        reason: String,
    },
    /// Runtime confirmed support.
    Available,
    /// Runtime confirmed the provider cannot be used.
    Unavailable {
        /// Explanation for the unavailable result.
        reason: String,
    },
}

/// Device class when the host can identify it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceClass {
    /// General-purpose CPU.
    Cpu,
    /// Integrated graphics processor.
    IntegratedGpu,
    /// Discrete graphics processor.
    DiscreteGpu,
    /// Non-GPU accelerator.
    Accelerator,
    /// Device class could not be identified.
    Unknown,
}
/// Best-effort compute device facts; missing values remain absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeDevice {
    /// Stable OS/provider scoped identifier.
    pub id: String,
    /// Device vendor, if identified.
    pub vendor: Option<String>,
    /// Device model/name, if identified.
    pub name: Option<String>,
    /// Device class.
    pub class: DeviceClass,
    /// Total memory in bytes, if reported.
    pub memory_bytes: Option<u64>,
}
/// One engine's view of a provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderCapability {
    /// Engine crate identifier.
    pub engine: String,
    /// Provider identifier.
    pub provider: ProviderId,
    /// Whether the adapter exists in this engine.
    pub implementation: ProviderStatus,
    /// Build and runtime state.
    pub runtime: ProviderStatus,
    /// Probe detail.
    pub details: Option<String>,
    /// Compatible device IDs.
    pub device_ids: Vec<String>,
}
/// Host and engine capability snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SystemCapabilities {
    /// OS name.
    pub os: String,
    /// CPU architecture.
    pub architecture: String,
    /// Identified compute devices.
    pub devices: Vec<ComputeDevice>,
    /// Engine/provider rows.
    pub providers: Vec<ProviderCapability>,
    /// State of host device enumeration. Empty devices with `Unknown` means no device probe ran.
    pub device_probe: ProviderStatus,
}

impl Default for ProviderStatus {
    fn default() -> Self {
        Self::Unknown {
            reason: "no capability probe has run".into(),
        }
    }
}

/// Return host OS and architecture facts without inferring accelerator availability.
pub fn host_capabilities() -> SystemCapabilities {
    SystemCapabilities {
        os: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        devices: Vec::new(),
        providers: Vec::new(),
        device_probe: ProviderStatus::Unknown {
            reason: "hardware and driver enumeration is not implemented in this facade".into(),
        },
    }
}
/// User's desired provider choice.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ProviderRequest {
    /// Select a provider explicitly.
    Explicit(ProviderId),
    /// Use deterministic CPU-first selection.
    #[default]
    Unspecified,
}
/// Provider-selection failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderSelectionError {
    /// Requested identifier.
    pub provider: ProviderId,
    /// Why the request could not be honored.
    pub reason: String,
}
impl fmt::Display for ProviderSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "provider {:?} cannot be selected: {}",
            self.provider, self.reason
        )
    }
}
impl std::error::Error for ProviderSelectionError {}
