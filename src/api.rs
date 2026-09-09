use crate::db;
use axum::{Json, extract::State};
use candle_core::Device;
use candle_transformers::models::bert::BertModel;
use rusqlite::Connection;
use serde::Deserialize;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

/// `/search` API json arguments
#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub top_k: usize,
}

/// required app state for RAG operations
pub struct AppState {
    pub model: BertModel,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub conn: Mutex<Connection>,
}

/// Performs search for API handler
pub async fn search_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> Json<Vec<(String, String, String, f32)>> {
    let conn = state.conn.lock().await;
    let results = db::search(
        &conn,
        &state.model,
        &state.tokenizer,
        &state.device,
        &req.query,
        req.top_k,
    )
    .unwrap();
    Json(results)
}
