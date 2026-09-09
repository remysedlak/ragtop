use anyhow::{Error, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

/// build HuggingFace bert model
pub fn load_bert_model(device: &Device, model: &str) -> Result<(BertModel, Tokenizer)> {
    // load the model from hf site
    let api = Api::new()?;
    // let repo = api.model("sentence-transformers/all-MiniLM-L6-v2".to_string());
    let repo = api.model(model.to_string());
    let config_filename = repo.get("config.json")?;
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let weights_filename = repo.get("model.safetensors")?;

    // create Config and Tokenizer
    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;
    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;
    let _ = tokenizer.with_truncation(None); // remove truncation, 128 ceiling caused problems with pdf chunking

    // build arguments for Bert Model
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };

    // load model
    let model = BertModel::load(vb, &config)?;

    Ok((model, tokenizer))
}

/// input one chunk of text and return embedding
pub fn embed(
    model: &BertModel,
    tokenizer: &Tokenizer,
    device: &Device,
    text: &str,
) -> Result<Vec<f32>> {
    let ec = tokenizer.encode(text, true).map_err(anyhow::Error::msg)?;
    let ids = ec.get_ids();
    let token_ids = Tensor::new(&ids[..], device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;
    let v = model.forward(&token_ids, &token_type_ids, None)?;
    let (_n, n_tokens, _hidden) = v.dims3()?;
    let pooled = (v.sum(1)? / (n_tokens as f64))?;
    let normalized = normalize_l2(&pooled)?;
    let flat: Vec<f32> = normalized.squeeze(0)?.to_vec1()?;
    Ok(flat)
}

/// L2-normalizes a tensor;
/// rescales each row vector so its Euclidean length is 1, without changing its direction.
pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
