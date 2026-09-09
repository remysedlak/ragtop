use candle_core::Device;

mod db;
mod extract;
mod hf;

pub fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let (model, tokenizer) = hf::load_model(&device)?;
    let conn = db::connection()?;
    db::create_document_table(&conn)?;
    db::create_chunks_table(&conn)?;
    let results = db::search(&conn, &model, &tokenizer, &device, "Javascript html", 5)?;
    for (source, unit, text, score) in results {
        println!("{score:.4} | {source} ({unit})\n{text}\n");
    }
    Ok(())
}
