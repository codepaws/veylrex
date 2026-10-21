//! Deterministic provider validation and CPU-first selection.
use crate::capability::{
    ProviderId, ProviderRequest, ProviderSelectionError, ProviderStatus, SystemCapabilities,
};

/// Resolve a provider request against one engine's capability rows.
pub fn select_provider(
    capabilities: &SystemCapabilities,
    request: &ProviderRequest,
) -> Result<ProviderId, ProviderSelectionError> {
    let requested = match request {
        ProviderRequest::Explicit(provider) => provider.clone(),
        ProviderRequest::Unspecified => ProviderId::Cpu,
    };
    let matching_rows = capabilities
        .providers
        .iter()
        .filter(|row| row.provider == requested)
        .collect::<Vec<_>>();
    if matching_rows.len() > 1 {
        return Err(ProviderSelectionError { provider: requested, reason: "provider occurs in multiple engines; select against an engine-specific capability view".into() });
    }
    let row = matching_rows.first().copied();
    let Some(row) = row else {
        return Err(ProviderSelectionError {
            provider: requested,
            reason: "no capability entry exists for this provider".into(),
        });
    };
    let reason = match &row.implementation {
        ProviderStatus::NotImplemented => Some("provider is not implemented".into()),
        ProviderStatus::NotCompiled => Some("provider is not compiled into this engine".into()),
        ProviderStatus::Unknown { reason } => Some(format!(
            "provider implementation status is unknown: {reason}"
        )),
        ProviderStatus::Unavailable { reason } => {
            Some(format!("provider implementation is unavailable: {reason}"))
        }
        ProviderStatus::Available => match &row.runtime {
            ProviderStatus::Available => None,
            ProviderStatus::Unavailable { reason } => Some(reason.clone()),
            ProviderStatus::NotImplemented => Some("provider is not implemented".into()),
            ProviderStatus::NotCompiled => Some("provider is not compiled into this engine".into()),
            ProviderStatus::Unknown { reason } => {
                Some(format!("runtime availability is unknown: {reason}"))
            }
        },
    };
    if let Some(reason) = reason {
        return Err(ProviderSelectionError {
            provider: requested,
            reason,
        });
    }
    Ok(requested)
}

/// Resolve a provider request for one selected engine.
pub fn select_provider_for_engine(
    capabilities: &SystemCapabilities,
    engine: &str,
    request: &ProviderRequest,
) -> Result<ProviderId, ProviderSelectionError> {
    let scoped = SystemCapabilities {
        providers: capabilities
            .providers
            .iter()
            .filter(|row| row.engine == engine)
            .cloned()
            .collect(),
        ..SystemCapabilities::default()
    };
    select_provider(&scoped, request)
}
