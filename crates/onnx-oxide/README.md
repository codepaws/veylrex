# onnx-oxide

Thin, explicit ONNX Runtime runner using the pinned `ort` 2.0.0-rc.13 crate. The default `runner` feature uses `load-dynamic`; it does not enable `download-binaries`, `copy-dylibs`, or model downloads. Callers provide the trusted ONNX Runtime shared-library path and model path.

The runner currently accepts one named f32 input tensor and returns owned f32 outputs. Enable `cuda` or `coreml` to compile those execution-provider adapters. Provider registration requires a successful model session; successful registration does not establish that every model operator ran on that provider. Runtime and per-model qualification remains outstanding.
