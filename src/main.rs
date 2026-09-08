use std::error::Error;

use candle_core::Device;
use walkdir::WalkDir;

use crate::db::{create_chunks_table, create_document_table, get_or_insert_document};

pub mod db;
pub mod extract;
pub mod hf;

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::Cpu;
    let (model, tokenizer) = hf::load_model(&device)?;

    // setup db
    let conn = db::connection()?;
    create_document_table(&conn)?;
    create_chunks_table(&conn)?;

    for entry in WalkDir::new("./test_folder") {
        let entry = entry?;
        if entry.file_type().is_file() {
            let source = entry
                .path()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let doc_id = get_or_insert_document(&conn, &source)?;
            match extract::get_chunks_from_file(&entry, &tokenizer) {
                Ok(chunks) => {
                    // println!("{:#?}", chunks);
                    for chunk in chunks {
                        let embedding = hf::embed(&model, &tokenizer, &device, &chunk.text)?;
                        db::insert_chunk(&conn, doc_id, &chunk.unit, &chunk.text, &embedding)?;
                        // now you have chunk.source, chunk.unit, chunk.text, and vector together
                        println!("{} -> {} dims", chunk.unit, embedding.len());
                    }
                }
                Err(e) => eprintln!("Failed on {}: {e}", entry.path().display()),
            }
        }
    }
    Ok(())
}
