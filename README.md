# ragtop

A local, offline semantic search tool for your school notes. Point it at a
folder of PDFs, Markdown notes, PowerPoint slides, or text files, and search
across all of them by meaning, not just keywords.

Everything runs locally — no API calls, no cloud services. Embeddings are
computed on your own machine using a small BERT-family model via
[Candle](https://github.com/huggingface/candle), and results are stored in a
local SQLite database.

## How it works

1. **Ingest**: walks a folder, extracts text from each file, splits it into
   overlapping token-sized chunks, filters out low-content noise (empty
   slides, bare tables, image placeholders), embeds each chunk into a vector,
   and stores everything in `notes.db`.
2. **Search**: embeds your query the same way, then ranks every stored chunk
   by cosine similarity to find the closest matches.

Re-running `ingest` on the same folder is safe — unchanged files are skipped
automatically, and files that have been edited since the last run are
re-processed (old chunks are replaced, not duplicated).

## Supported file types

- `.pdf`
- `.md`
- `.pptx`
- `.txt`

Other file types are ignored during ingestion.

## Usage

Build the release binary first (release mode matters — embedding is
significantly slower in a debug build):

```
cargo build --release
```

### Ingest a folder

```
./target/release/ragtop ingest <path>
```

Example:

```
./target/release/ragtop ingest ./my_notes
```

This walks `<path>` recursively, processing every supported file it finds.
On first run, every file is embedded and stored. On later runs, only new or
modified files are re-processed.

### Search

```
./target/release/ragtop search "<query>" --top-k <n>
```

`--top-k` (or `-t`) is optional and defaults to `5`.

Example:

```
./target/release/ragtop search "how does the linux kernel handle syscalls" --top-k 10
```

Each result shows a similarity score, the source file and location within it
(page number, slide number, or section heading), and the matching text.

### Reset Database

To easily reset the database, run:

```
./target/release/ragtop reset
```

### Help

```
./target/release/ragtop --help
./target/release/ragtop ingest --help
./target/release/ragtop search --help
```

## Notes on search quality

- Full-sentence, question-style queries generally retrieve better than
  single keywords — the embedding model is trained on sentence-length
  inputs, not bare tags.
- Slide decks that are mostly diagrams, tables, or images will have little
  or no extractable text; this is a limitation of the source material, not
  the pipeline.
- The database (`notes.db`) is created in the current working directory. Run
  the binary from the same directory each time, or the CLI will look for a
  different `notes.db` depending on where you launch it from.

## Requirements

- Rust (edition 2024)
- Internet access on first run only, to download the embedding model from
  Hugging Face (cached locally afterward)
