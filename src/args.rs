use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(author, version, about)]
pub struct RagArgs {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// use binary to upload RAG documents
    Ingest { path: String },
    /// use binary to search RAG database
    Search {
        query: String,
        #[arg(short, long, default_value_t = 5)]
        top_k: usize,
    },
    /// launch an HTTP server on somes port
    Serve {
        #[arg(short, long, default_value_t = 3000)]
        port: u16,
    },
    /// Reset the database
    Reset,
}
