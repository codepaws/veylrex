use crate::{GenerationOptions, GenerationOutput, Provider};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
};
use std::{error::Error as StdError, num::NonZeroU32, path::Path, sync::OnceLock};

static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();

fn backend() -> Result<&'static LlamaBackend, Box<dyn StdError>> {
    BACKEND
        .get_or_init(|| LlamaBackend::init().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| error.clone().into())
}

pub(super) fn capabilities() -> Vec<crate::ProviderCapability> {
    let backend = backend();
    [
        Provider::Cpu,
        Provider::Cuda,
        Provider::Vulkan,
        Provider::Metal,
    ]
    .into_iter()
    .map(|provider| {
        let compiled = match provider {
            Provider::Cpu => true,
            Provider::Cuda => cfg!(feature = "cuda"),
            Provider::Vulkan => cfg!(feature = "vulkan"),
            Provider::Metal => cfg!(feature = "metal"),
        };
        let (available, reason) = if !compiled {
            (
                Some(false),
                Some("provider backend feature is not compiled".into()),
            )
        } else {
            match &backend {
                Err(error) => (None, Some(error.to_string())),
                Ok(_backend) if provider == Provider::Cpu => (Some(true), None),
                Ok(backend) if backend.supports_gpu_offload() => (
                    Some(true),
                    Some(
                        "llama.cpp reports GPU offload support; no model session has been verified"
                            .into(),
                    ),
                ),
                Ok(_) => (
                    Some(false),
                    Some("llama.cpp reports no compatible GPU backend at runtime".into()),
                ),
            }
        };
        crate::ProviderCapability {
            provider,
            compiled,
            available,
            reason,
        }
    })
    .collect()
}

pub(super) fn generate(
    model_path: &Path,
    prompt: &str,
    options: GenerationOptions,
    provider: Provider,
) -> Result<GenerationOutput, Box<dyn StdError>> {
    validate_provider(provider)?;
    if prompt.is_empty() {
        return Err("prompt must not be empty".into());
    }
    if !model_path.is_file() {
        return Err(format!("GGUF model file does not exist: {}", model_path.display()).into());
    }
    let backend = backend()?;
    if provider != Provider::Cpu && !backend.supports_gpu_offload() {
        return Err("llama.cpp has no usable GPU backend on this system".into());
    }

    let mut model_params = LlamaModelParams::default();
    if provider != Provider::Cpu {
        model_params = model_params.with_n_gpu_layers(u32::MAX);
    }
    let model = LlamaModel::load_from_file(backend, model_path, &model_params)?;
    let context_size =
        NonZeroU32::new(options.context_size).ok_or("context_size must be positive")?;
    let context_params = LlamaContextParams::default()
        .with_n_ctx(Some(context_size))
        .with_n_threads(options.threads as i32)
        .with_n_threads_batch(options.threads as i32);
    let mut context = model.new_context(backend, context_params)?;
    let prompt_tokens = model.vocab().tokenize(prompt.as_bytes(), true, true);
    if prompt_tokens.is_empty() {
        return Err("prompt tokenization produced no tokens".into());
    }
    if prompt_tokens.len() >= options.context_size as usize {
        return Err(format!(
            "prompt has {} tokens but context size is {}",
            prompt_tokens.len(),
            options.context_size
        )
        .into());
    }

    let mut batch = LlamaBatch::new(prompt_tokens.len(), 1);
    let last_prompt_index = i32::try_from(prompt_tokens.len() - 1)?;
    for (index, token) in prompt_tokens.iter().enumerate() {
        batch.add(
            *token,
            i32::try_from(index)?,
            &[0],
            i32::try_from(index)? == last_prompt_index,
        )?;
    }
    context.decode(&mut batch)?;

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut generated = Vec::with_capacity(options.max_tokens as usize);
    for _ in 0..options.max_tokens {
        let token = sampler.sample(&context, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.vocab().is_eog(token) {
            break;
        }
        generated.push(token);
        batch.clear();
        let position = i32::try_from(prompt_tokens.len() + generated.len() - 1)?;
        batch.add(token, position, &[0], true)?;
        context.decode(&mut batch)?;
    }
    let text = model.vocab().detokenize(&generated, true, true);
    Ok(GenerationOutput {
        text: String::from_utf8_lossy(&text).into_owned(),
        provider,
    })
}

fn validate_provider(provider: Provider) -> Result<(), Box<dyn StdError>> {
    let enabled_gpu_backends = usize::from(cfg!(feature = "cuda"))
        + usize::from(cfg!(feature = "vulkan"))
        + usize::from(cfg!(feature = "metal"));
    let compiled = match provider {
        Provider::Cpu => true,
        Provider::Cuda => cfg!(feature = "cuda"),
        Provider::Vulkan => cfg!(feature = "vulkan"),
        Provider::Metal => cfg!(feature = "metal"),
    };
    if !compiled {
        return Err(format!("llama.cpp {:?} backend was not compiled in", provider).into());
    }
    if provider != Provider::Cpu && enabled_gpu_backends != 1 {
        return Err("provider-specific llama.cpp execution requires exactly one GPU backend feature to be enabled".into());
    }
    Ok(())
}
