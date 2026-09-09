use crate::hf;
use candle_core::Device;
use candle_transformers::models::bert::BertModel;
use rusqlite::{Connection, Error};
use tokenizers::Tokenizer;

/// Return connection to SQLite database
pub fn connection() -> Result<Connection, Error> {
    let conn = Connection::open("notes.db");
    conn
}

// ID : PK
// SOURCE
//
pub fn create_document_table(conn: &Connection) -> Result<usize, Error> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            modified_at INTEGER,
            UNIQUE(source)
        );",
        (),
    )
}
// ID : PK
// DOCUMENT_ID : FK
// UNIT
// TEXT
// EMBEDDING
//
pub fn create_chunks_table(conn: &Connection) -> Result<usize, Error> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id),
            unit TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL
        );",
        (),
    )
}

/// append or return existing document using source
pub fn get_or_insert_document(conn: &Connection, source: &str) -> Result<i64, Error> {
    conn.execute(
        "INSERT OR IGNORE INTO documents (source) VALUES (?1)",
        (source,),
    )?;

    conn.query_row(
        "SELECT id FROM documents WHERE source = ?1",
        (source,),
        |row| row.get(0),
    )
}

/// turn f32 values into BLOB bytes
fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}
/// turn BLOB bytes into f32 values
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// enter one chunk into the database
pub fn insert_chunk(
    conn: &Connection,
    document_id: i64,
    unit: &str,
    text: &str,
    embedding: &[f32],
) -> Result<usize, Error> {
    let bytes = f32_vec_to_bytes(embedding);
    conn.execute(
        "INSERT INTO chunks (document_id, unit, text, embedding) VALUES (?1, ?2, ?3, ?4)",
        (document_id, unit, text, &bytes),
    )
}

/// return all chunks for brute forcing
pub fn get_all_chunks(conn: &Connection) -> Result<Vec<(String, String, String, Vec<f32>)>, Error> {
    let mut stmt = conn.prepare(
        "SELECT documents.source, chunks.unit, chunks.text, chunks.embedding
         FROM chunks
         JOIN documents ON chunks.document_id = documents.id",
    )?;

    let rows = stmt.query_map((), |row| {
        let source: String = row.get(0)?;
        let unit: String = row.get(1)?;
        let text: String = row.get(2)?;
        let blob: Vec<u8> = row.get(3)?;
        let embedding = bytes_to_f32_vec(&blob);
        Ok((source, unit, text, embedding))
    })?;

    rows.collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

/// brute force search function
/// return top_k cosine simularity results for a query
pub fn search(
    conn: &Connection,
    model: &BertModel,
    tokenizer: &Tokenizer,
    device: &Device,
    query: &str,
    top_k: usize,
) -> anyhow::Result<Vec<(String, String, String, f32)>> {
    let query_vec = hf::embed(model, tokenizer, device, query)?;
    eprintln!("query: {query} | first 5 dims: {:?}", &query_vec[..5]);
    let all_chunks = get_all_chunks(conn)?;

    let mut scored: Vec<(String, String, String, f32)> = all_chunks
        .into_iter()
        .map(|(source, unit, text, emb)| {
            let score = cosine_similarity(&query_vec, &emb);
            (source, unit, text, score)
        })
        .collect();

    scored.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    scored.truncate(top_k);

    Ok(scored)
}

pub fn get_document_modified(conn: &Connection, source: &str) -> Result<Option<i64>, Error> {
    match conn.query_row(
        "SELECT modified_at FROM documents WHERE source = ?1",
        (source,),
        |row| row.get(0),
    ) {
        Ok(modified) => Ok(modified),
        Err(Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// when a document that already existed is ingested, modify the old version instead of duplication
pub fn update_document_modified(
    conn: &Connection,
    document_id: i64,
    modified: i64,
) -> Result<usize, Error> {
    conn.execute(
        "UPDATE documents SET modified_at = ?1 WHERE id = ?2",
        (modified, document_id),
    )
}

/// delete all chunks for one document source
pub fn delete_chunks_for_document(conn: &Connection, document_id: i64) -> Result<usize, Error> {
    conn.execute("DELETE FROM chunks WHERE document_id = ?1", (document_id,))
}
