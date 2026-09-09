use crate::api::{AppState, handler};
use crate::args::{Commands, RagArgs};
use crate::db::{create_chunks_table, create_document_table};
use crate::extract::is_degenerate;
use axum::{Router, routing::get};
use candle_core::Device;
use clap::Parser;
use tower_http::services::ServeDir;

use std::sync::Arc;

use tokio::sync::Mutex;
use walkdir::WalkDir;

mod api;
mod args;
mod db;
mod extract;
mod hf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = RagArgs::parse();

    let device = Device::Cpu;
    let (model, tokenizer) = hf::load_bert_model(&device)?;
    let conn = db::connection()?;
    create_document_table(&conn)?;
    create_chunks_table(&conn)?;

    match args.command {
        Commands::Serve { port } => {
            // build our application with a single route

            let shared_state = Arc::new(AppState {
                model,
                tokenizer,
                device,
                conn: Mutex::new(conn),
            });
            let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
                .await
                .unwrap();
            let app = Router::new()
                .route("/search", axum::routing::post(api::search_handler))
                .with_state(shared_state)
                .fallback_service(ServeDir::new("static"));

            axum::serve(listener, app).await.unwrap();
        }
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
