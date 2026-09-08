use anyhow;
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::api::sync::Api;
use std::error::Error;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn Error>> {
    let api = Api::new()?;
    let device = Device::Cpu;
    let repo = api.model("sentence-transformers/all-MiniLM-L6-v2".to_string());
    let config_filename = repo.get("config.json")?;
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let weights_filename = repo.get("model.safetensors")?;

    println!("{:?}", config_filename);
    println!("{:?}", tokenizer_filename);
    println!("{:?}", weights_filename);

    let config = std::fs::read_to_string(config_filename)?;
    let mut config: Config = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };

    let model = BertModel::load(vb, &config)?;

    Ok(())
}
