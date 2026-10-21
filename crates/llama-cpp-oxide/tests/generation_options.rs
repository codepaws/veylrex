//! Native-free LLM request validation tests.

use llama_cpp_oxide::GenerationOptions;

#[test]
fn generation_options_reject_zero_tokens_and_context() {
    assert!(GenerationOptions::new(0, 128, 4).is_err());
    assert!(GenerationOptions::new(4, 0, 4).is_err());
}

#[test]
fn generation_options_accept_positive_limits() {
    let options = GenerationOptions::new(16, 2048, 4).unwrap();
    assert_eq!(options.max_tokens, 16);
    assert_eq!(options.context_size, 2048);
    assert_eq!(options.threads, 4);
}
