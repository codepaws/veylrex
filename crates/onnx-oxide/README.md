# onnx-oxide

`onnx-oxide` is a qualification-stage candidate for a thin Veylrex adapter around [`ort`](https://github.com/pykeio/ort). The preferred direction is to reuse `ort`'s safe API and raw `ort-sys` surface rather than create a second ONNX Runtime binding. This crate should remain only if Veylrex needs a small, well-defined adapter for its task contracts or lifetime-free ownership model; otherwise Veylrex can depend on `ort` directly.

Before choosing either integration shape, qualify an exact `ort` version for Windows x86-64, Linux x86-64, and Linux ARM64. Disable default features that download binaries or copy runtime libraries implicitly, test explicit dynamic loading, verify raw API coverage through `ort::sys`, and confirm provider behavior with representative embedding and reranking models. Provider support is reported only from qualified runtime builds; a Cargo feature alone does not claim that a GPU provider is usable.

Any original adapter code is dual-licensed under MIT OR Apache-2.0. `ort`, ONNX Runtime, and their transitive dependencies retain their own licenses and notices. See the integration target spec for the qualification gate and fallback criteria.
