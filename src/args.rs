use clap::{Parser, Subcommand};

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
    Serve {
        #[arg(short, long, default_value_t = 3000)]
        port: u16,
    },
}
