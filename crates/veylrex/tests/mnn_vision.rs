//! Contract checks for Veylrex's MNN vision facade.
#![cfg(feature = "mnn")]
use veylrex::{ProviderId, ProviderRequest, VisionInput, VisionLayout, mnn};

#[test]
fn mnn_provider_inventory_reports_native_provider_state_and_unknown_devices() {
    let snapshot = veylrex::system_capabilities();
    assert!(!snapshot.os.is_empty() && !snapshot.architecture.is_empty());
    assert!(matches!(
        snapshot.device_probe,
        veylrex::ProviderStatus::Unknown { .. }
    ));
    let entries = snapshot.providers;
    assert!(entries.iter().any(|entry| entry.provider == ProviderId::Cpu
        && entry.runtime == veylrex::ProviderStatus::Available));
    assert!(
        entries
            .iter()
            .any(|entry| entry.provider == ProviderId::Cuda
                && matches!(entry.implementation, veylrex::ProviderStatus::NotCompiled))
    );
}

#[test]
fn rejects_inconsistent_input_before_loading_model() {
    let input = VisionInput {
        model: "missing-model.mnn".into(),
        data: vec![0.0; 2],
        shape: vec![1, 3],
        layout: VisionLayout::Nchw,
        input_name: None,
        output_names: vec![],
        provider: ProviderRequest::Unspecified,
    };
    let err = mnn::run(input).unwrap_err().to_string();
    assert!(err.contains("shape requires 3"), "{err}");
}
