//! Native-free tests for ONNX Runtime capability reporting.

#![cfg(feature = "onnx")]

use veylrex::{ProviderId, ProviderStatus};

#[test]
fn onnx_rows_distinguish_uncompiled_accelerators_from_unknown_runtime_state() {
    let capabilities = veylrex::system_capabilities();
    let cpu = capabilities
        .providers
        .iter()
        .find(|entry| entry.engine == "onnx-oxide" && entry.provider == ProviderId::Cpu)
        .unwrap();
    assert!(matches!(cpu.implementation, ProviderStatus::Available));
    assert!(matches!(cpu.runtime, ProviderStatus::Unknown { .. }));

    let cuda = capabilities
        .providers
        .iter()
        .find(|entry| entry.engine == "onnx-oxide" && entry.provider == ProviderId::Cuda)
        .unwrap();
    if cfg!(feature = "onnx-cuda") {
        assert!(matches!(cuda.implementation, ProviderStatus::Available));
    } else {
        assert!(matches!(cuda.implementation, ProviderStatus::NotCompiled));
    }

    assert!(
        veylrex::select_provider_for_engine(
            &capabilities,
            "onnx-oxide",
            &veylrex::ProviderRequest::Explicit(ProviderId::Cpu)
        )
        .is_err()
    );
}
