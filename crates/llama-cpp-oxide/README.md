# llama.cpp-oxide

`llama-cpp-oxide` is the dedicated Rust boundary for llama.cpp. It owns the pinned upstream revision, binding generation, dynamic loading, managed native handles, safe text-generation APIs, and target-specific native artifacts for llama.cpp.

The crate is intended to provide broad access to the pinned upstream public C API plus an idiomatic, `Result`-based Rust layer for common model-loading and generation workflows. It does not build an application server, manage RAG collections, or hide engine-specific capabilities behind lossy abstractions.

Original Rust code is dual-licensed under MIT OR Apache-2.0. llama.cpp and its dependencies retain their own licenses; downstream distributions must preserve all required notices. See the engine target spec for the complete contract.
