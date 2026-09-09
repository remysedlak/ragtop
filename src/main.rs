use candle_core::Device;
use clap::Parser;
use walkdir::WalkDir;

mod db;
mod extract;
mod hf;

use crate::{
    db::{create_chunks_table, create_document_table},
    extract::is_degenerate,
};

#[derive(Debug, Parser)]
#[command(author, version, about)]
pub struct RagArgs {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, clap::Subcommand)]
pub enum Commands {
    Ingest {
        path: String,
    },
    Search {
        query: String,
        #[arg(short, long, default_value_t = 5)]
        top_k: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let args = RagArgs::parse();

    let device = Device::Cpu;
    let (model, tokenizer) = hf::load_bert_model(&device)?;
    let conn = db::connection()?;
    create_document_table(&conn)?;
    create_chunks_table(&conn)?;

    match args.command {
        Commands::Ingest { path } => {
            for entry in WalkDir::new(&path) {
                let entry = entry?;
                if entry.file_type().is_file() {
                    let source = entry.path().to_string_lossy().to_string();
                    let modified = entry
                        .metadata()?
                        .modified()?
                        .duration_since(std::time::UNIX_EPOCH)?
                        .as_secs() as i64;

                    let existing = db::get_document_modified(&conn, &source)?;
                    if existing == Some(modified) {
                        println!("SKIP (unchanged): {source}");
                        continue;
                    }

                    let doc_id = db::get_or_insert_document(&conn, &source)?;
                    db::update_document_modified(&conn, doc_id, modified)?;
                    db::delete_chunks_for_document(&conn, doc_id)?;

                    match extract::get_chunks_from_file(&entry, &tokenizer) {
                        Ok(chunks) => {
                            for chunk in chunks {
                                if chunk.text.trim().is_empty() || is_degenerate(&chunk.text) {
                                    continue;
                                }
                                let embedding =
                                    hf::embed(&model, &tokenizer, &device, &chunk.text)?;
                                db::insert_chunk(
                                    &conn,
                                    doc_id,
                                    &chunk.unit,
                                    &chunk.text,
                                    &embedding,
                                )?;
                            }
                        }
                        Err(e) => eprintln!("Failed on {}: {e}", entry.path().display()),
                    }
                }
            }
        }
        Commands::Search { query, top_k } => {
            let results = db::search(&conn, &model, &tokenizer, &device, &query, top_k)?;
            for (source, unit, text, score) in results {
                println!("{score:.4} | {source} ({unit})\n{text}\n");
            }
        }
    }
    Ok(())
}
