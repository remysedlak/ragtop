use rusqlite::{Connection, Error};

pub fn connection() -> Result<Connection, Error> {
    let conn = Connection::open("notes.db");
    conn
}

pub fn create_document_table(conn: &Connection) -> Result<usize, Error> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            UNIQUE(source)
        );",
        (),
    )
}

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

fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

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
