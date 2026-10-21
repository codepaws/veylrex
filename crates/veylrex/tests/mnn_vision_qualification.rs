//! Model-backed comparison of the Veylrex adapter and direct mnn-oxide CPU inference.
#![cfg(feature = "mnn")]

use mnn_oxide::{BackendType, Interpreter, ScheduleConfig};
use veylrex::{ProviderId, ProviderRequest, VisionInput, VisionLayout, mnn};

#[test]
#[ignore = "requires the pinned PP-LCNet model at MNN_TEST_MODEL"]
fn veylrex_pp_lcnet_outputs_match_direct_mnn_cpu_for_nonzero_input() {
    let model =
        std::env::var("MNN_TEST_MODEL").expect("set MNN_TEST_MODEL to the pinned PP-LCNet fixture");
    let shape = vec![1, 3, 80, 160];
    let input = (0..(shape.iter().product::<i32>() as usize))
        .map(|index| ((index % 29) as f32 - 14.0) / 14.0)
        .collect::<Vec<_>>();

    let interpreter = Interpreter::from_file(&model).unwrap();
    let mut session = interpreter
        .create_session(ScheduleConfig::new().backend(BackendType::CPU))
        .unwrap();
    let tensor = session.get_input(None).unwrap();
    assert_eq!(tensor.info().unwrap().shape, shape);
    tensor.write(&input).unwrap();
    session.run().unwrap();
    let direct = session.get_output(None).unwrap().read::<f32>().unwrap();

    let adapted = mnn::run(VisionInput {
        model: model.into(),
        data: input,
        shape,
        layout: VisionLayout::Nchw,
        input_name: None,
        output_names: vec![],
        provider: ProviderRequest::Explicit(ProviderId::Cpu),
    })
    .unwrap();
    assert_eq!(adapted.outputs.len(), 1);
    assert_eq!(adapted.requested_provider, ProviderId::Cpu);
    assert_eq!(adapted.native_provider, ProviderId::Cpu);
    assert_eq!(adapted.outputs[0].shape, vec![1, 2]);
    assert_eq!(adapted.outputs[0].data.len(), direct.len());
    for (actual, expected) in adapted.outputs[0].data.iter().zip(direct) {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "adapter {actual} != direct {expected}"
        );
    }
}
