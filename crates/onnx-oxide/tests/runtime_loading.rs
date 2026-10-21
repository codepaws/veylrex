//! ONNX runtime loading failure tests do not need native runtime binaries.

use onnx_oxide::{Provider, Runner};

#[test]
fn missing_runtime_library_returns_error_without_downloading_runtime() {
    let runtime = std::env::temp_dir().join(format!(
        "veylrex-missing-onnxruntime-{}.dll",
        std::process::id()
    ));
    let model = runtime.with_extension("onnx");
    let error = Runner::load(&runtime, &model, Provider::Cpu)
        .err()
        .expect("missing ONNX Runtime should fail");
    assert!(!runtime.exists());
    assert!(!error.to_string().is_empty());
}
