//! Provider capability and selection contract tests.
use veylrex::*;

fn row(
    provider: ProviderId,
    implementation: ProviderStatus,
    runtime: ProviderStatus,
) -> ProviderCapability {
    ProviderCapability {
        engine: "fixture".into(),
        provider,
        implementation,
        runtime,
        details: None,
        device_ids: vec![],
    }
}
fn caps(rows: Vec<ProviderCapability>) -> SystemCapabilities {
    SystemCapabilities {
        os: "test".into(),
        architecture: "test".into(),
        devices: vec![],
        providers: rows,
        device_probe: ProviderStatus::Unknown {
            reason: "test".into(),
        },
    }
}

#[test]
fn explicit_provider_requires_implemented_compiled_and_runtime_available() {
    for (implementation, runtime, expected) in [
        (
            ProviderStatus::NotImplemented,
            ProviderStatus::Available,
            "not implemented",
        ),
        (
            ProviderStatus::NotCompiled,
            ProviderStatus::Available,
            "not compiled",
        ),
        (
            ProviderStatus::Available,
            ProviderStatus::Unknown {
                reason: "probe unavailable".into(),
            },
            "unknown",
        ),
        (
            ProviderStatus::Available,
            ProviderStatus::Unavailable {
                reason: "no driver".into(),
            },
            "no driver",
        ),
    ] {
        let c = caps(vec![row(ProviderId::Cuda, implementation, runtime)]);
        let e = select_provider(&c, &ProviderRequest::Explicit(ProviderId::Cuda)).unwrap_err();
        assert!(e.reason.contains(expected), "{}", e.reason);
    }
    let c = caps(vec![row(
        ProviderId::Cuda,
        ProviderStatus::Available,
        ProviderStatus::Available,
    )]);
    assert_eq!(
        select_provider(&c, &ProviderRequest::Explicit(ProviderId::Cuda)),
        Ok(ProviderId::Cuda)
    );
}

#[test]
fn unspecified_provider_selects_cpu_independent_of_device_order() {
    let mut c = caps(vec![
        row(
            ProviderId::Vulkan,
            ProviderStatus::Available,
            ProviderStatus::Available,
        ),
        row(
            ProviderId::Cpu,
            ProviderStatus::Available,
            ProviderStatus::Available,
        ),
    ]);
    c.devices.reverse();
    assert_eq!(
        select_provider(&c, &ProviderRequest::Unspecified),
        Ok(ProviderId::Cpu)
    );
}

#[test]
fn missing_cpu_and_provider_rows_are_errors() {
    let c = caps(vec![]);
    assert!(
        select_provider(&c, &ProviderRequest::Unspecified)
            .unwrap_err()
            .reason
            .contains("no capability")
    );
    assert!(select_provider(&c, &ProviderRequest::Explicit(ProviderId::CoreMl)).is_err());
}

#[test]
fn engine_scoped_selection_does_not_borrow_another_engines_cpu() {
    let c = caps(vec![
        row(
            ProviderId::Cpu,
            ProviderStatus::Available,
            ProviderStatus::Available,
        ),
        ProviderCapability {
            engine: "other".into(),
            provider: ProviderId::Cpu,
            implementation: ProviderStatus::NotCompiled,
            runtime: ProviderStatus::Unknown {
                reason: "not built".into(),
            },
            details: None,
            device_ids: vec![],
        },
    ]);
    assert_eq!(
        select_provider_for_engine(&c, "fixture", &ProviderRequest::Unspecified),
        Ok(ProviderId::Cpu)
    );
    assert!(
        select_provider_for_engine(&c, "other", &ProviderRequest::Unspecified)
            .unwrap_err()
            .reason
            .contains("not compiled")
    );
    assert!(
        select_provider(&c, &ProviderRequest::Unspecified)
            .unwrap_err()
            .reason
            .contains("multiple engines")
    );
}
