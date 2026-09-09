use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(author, version, about)]
pub struct RagArgs {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Ingest all files in a folder into the database
    Ingest {
        /// Path to the folder to ingest
        path: String,
    },
    /// Search stored chunks for a query
    Search {
        /// The search query
        query: String,
        #[arg(short, long, default_value_t = 5)]
        top_k: usize,
    },
}
