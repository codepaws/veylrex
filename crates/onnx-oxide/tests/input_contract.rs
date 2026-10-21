//! Native-free ONNX input contract tests.

use onnx_oxide::TensorInput;

#[test]
fn tensor_input_requires_positive_dimensions_and_matching_data_length() {
    let error = TensorInput::new("input", vec![1, 2], vec![0.0; 3]).unwrap_err();
    assert!(error.to_string().contains("shape requires 2"));
}

#[test]
fn tensor_input_accepts_exact_f32_shape() {
    let input = TensorInput::new("input", vec![1, 2], vec![3.0, 4.0]).unwrap();
    assert_eq!(input.name, "input");
    assert_eq!(input.shape, [1, 2]);
    assert_eq!(input.data, [3.0, 4.0]);
}
