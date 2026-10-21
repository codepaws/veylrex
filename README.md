# Veylrex

Veylrex is an in-process Rust model-runner platform. It brings task-oriented interfaces over independently maintained native runtime wrappers, so applications can select the engines they need without owning each runtime's FFI, native build, and packaging work.

The repository is a Cargo workspace. Each engine adapter owns its native API boundary, while the `veylrex` crate provides the application-facing orchestration layer. Engine dependencies remain optional; adding Veylrex does not implicitly install every native runtime.

## Workspace crates

| Crate | Responsibility | Status |
| --- | --- | --- |
| [`veylrex`](crates/veylrex/README.md) | Provider capability contracts, deterministic selection, and preprocessed MNN vision inference | Initial implementation |
| [`llama-cpp-oxide`](crates/llama-cpp-oxide/README.md) | Rust interface and native packaging for llama.cpp | Scaffold; runner not implemented |
| [`onnx-oxide`](crates/onnx-oxide/README.md) | ONNX Runtime f32 tensor runner with explicit runtime loading | Initial CPU runner; CUDA/CoreML build options |

MNN is consumed through the default-enabled, disableable `mnn` feature from the pinned `codepaws/mnn-oxide` Git revision in the crate manifest. That repository owns bindings, native builds, and packaging. A consumer must configure an explicit MNN native artifact; Cargo does not fetch or build the MNN runtime automatically. Veylrex accepts preprocessed f32 image tensors and returns raw output tensors. It has no image decoding, preprocessing, or OCR workflow API.

The shared provider API distinguishes implementation, build, and runtime state. MNN CPU is selectable when its native creator and runtime probe succeed. Accelerator adapter features can be enabled with `mnn-cuda`, `mnn-vulkan`, `mnn-metal`, or `mnn-opencl`; each must pass native capability probing before selection. ONNX Runtime is pinned to `ort` 2.0.0-rc.13 with binary download and copy features disabled. `onnx-cuda` and `onnx-coreml` enable the corresponding runtime APIs; callers supply an explicit ONNX Runtime shared-library path, and provider status stays unknown until runtime and session evidence exists. Explicit unknown providers are rejected. An omitted provider selects CPU. The llama.cpp crate remains a scaffold and does not yet implement inference or provider backends.

## Design principles

- Original Rust code in each crate is dual-licensed under MIT OR Apache-2.0. Native runtimes and other third-party material retain their own licenses and notices.
- Runtime crates own their bindings, native loading, safe handles, capability reporting, and native release artifacts.
- Native versions, source revisions, build options, and artifact checksums are pinned and recorded for reproducible releases.
- Ordinary Cargo builds do not silently fetch native runtimes, model files, or toolchains. Native artifacts are built and published as explicit, versioned target releases.
- Applications use in-process runners. A model-serving daemon, RAG pipeline, vector database, UI, and PDF processing are outside this project.
- Threading, callbacks, initialization, and lifetime guarantees are defined per runtime from its actual API contract; the common facade does not claim one concurrency model for all engines.

## Supported targets

The initial qualification targets are Windows x86-64, Linux x86-64, and Linux ARM64. A target is supported by an engine only after its native artifact, loader behavior, representative models, and acceptance tests pass on that target. Accelerator providers are qualified separately from CPU support.

## Repository setup

The repository already has Git initialized. The target specifications are stored locally under the gitignored `docs/superpowers/specs/` folder; implementation plans will live under `docs/superpowers/plans/`.

See `docs/superpowers/specs/2026-10-18-veylrex-orchestrator-target-spec.md` for the orchestration contract and the engine-specific specifications alongside it for each native boundary.
