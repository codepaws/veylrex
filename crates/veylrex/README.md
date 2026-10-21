# veylrex

`veylrex` is the application-facing in-process orchestrator for optional model-runner crates. It provides shared task contracts and explicit engine selection while leaving native API access, builds, and packaging to each engine's own crate.

The facade should stay task-oriented: embedding, reranking, and text generation are separate capabilities, and no engine is assumed to implement every task. Engine selection and device choice are explicit and capability-checked. Applications can use an engine crate directly when they need native-specific controls.

The default feature enables the optional Git-pinned `mnn-oxide` dependency; disable it with `--no-default-features`. Native MNN libraries and headers must be supplied explicitly to `mnn-oxide`. The facade currently accepts only already-preprocessed f32 image tensors and exposes raw output tensors. It does not decode images or provide an OCR workflow API.

Provider state is available through `SystemCapabilities` and is reported per engine. Explicit requests require both an implemented provider and a confirmed runtime availability result; unknown results are not silently replaced. Unspecified selection chooses CPU deterministically. MNN accelerator compile features are opt-in and do not prove native runtime support. ONNX Runtime and llama.cpp remain scaffolds without inference runners.

Querying `system_capabilities()` initializes each enabled MNN provider runtime to detect missing providers and CPU fallback. This may load provider drivers; it does not run a model or establish model/operator coverage. Host GPU/device enumeration is still reported as unknown until a platform inventory probe is implemented.

This crate is dual-licensed under MIT OR Apache-2.0. Third-party runtime licenses and notices remain with their respective engine crates and native distributions. See the workspace target spec for API, safety, and compatibility requirements.
