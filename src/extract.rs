use pptx_to_md::{ParserConfig, PresentationContainer};
use std::{error::Error, fs};
use tokenizers::Tokenizer;
use walkdir::DirEntry;

/// A chunk is a piece of a document used for RAG comparison
#[derive(Debug)]
pub struct Chunk {
    pub source: String,
    pub unit: String,
    pub text: String,
}

/// Splits text into ~chunk_words-sized pieces with overlap, tagging each
/// with the same source/unit (since they all came from the same segment).
fn chunk_string(
    source: &str,
    unit: &str,
    text: &str,
    tokenizer: &Tokenizer,
    chunk_tokens: usize,
    overlap_tokens: usize,
) -> Vec<Chunk> {
    // run the tokenizer on the full text
    let encoding = match tokenizer.encode(text, false) {
        Ok(e) => e,
        Err(_) => return vec![],
    };
    let ids = encoding.get_ids();
    let offsets = encoding.get_offsets();

    if ids.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    // chunk until out of string
    loop {
        let end = (start + chunk_tokens).min(ids.len());

        let char_start = offsets[start].0;
        let char_end = offsets[end - 1].1;

        chunks.push(Chunk {
            source: source.to_string(),
            unit: unit.to_string(),
            text: text[char_start..char_end].to_string(),
        });

        if end == ids.len() {
            break;
        }
        start = end - overlap_tokens;
    }

    chunks
}

/// attempt to remove non-textual elements from chunks
fn strip_noise(text: &str) -> String {
    let mut result = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[Image unavailable:")
            || trimmed.starts_with("<!-- Unsupported slide element:")
            || trimmed.starts_with("**Source:")
        {
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }
    result
}

pub fn is_degenerate(text: &str) -> bool {
    let stripped = strip_noise(text);
    let real_words = stripped
        .split_whitespace()
        .filter(|w| w.chars().any(|c| c.is_alphabetic()))
        .count();
    real_words < 15
}

/// Extracts a file's text, splits it into structural units (page/slide/
/// header/paragraph depending on file type), and chunks each unit.
pub fn get_chunks_from_file(
    entry: &DirEntry,
    tokenizer: &Tokenizer,
) -> Result<Vec<Chunk>, Box<dyn Error>> {
    let path = entry.path();
    let ext = path.extension().and_then(|e| e.to_str());
    let source = path.file_name().unwrap().to_string_lossy().to_string();

    match ext {
        Some("txt") => {
            // get text from file
            let contents = fs::read_to_string(path)?;
            let mut chunks = Vec::new();
            for (i, para) in contents
                .split("\n\n")
                .filter(|p| !p.trim().is_empty())
                .enumerate()
            {
                let unit = format!("para_{}", i + 1);
                chunks.extend(chunk_string(
                    &source,
                    &unit,
                    para.trim(),
                    tokenizer,
                    200,
                    40,
                ));
            }
            Ok(chunks)
        }
        Some("pdf") => {
            let bytes = fs::read(path)?;
            let pages = pdf_extract::extract_text_from_mem_by_pages(&bytes)?;
            let mut full_text = String::new();
            let mut page_boundaries = Vec::new(); // (start_char, page_number)

            for (i, page_text) in pages.iter().enumerate() {
                page_boundaries.push((full_text.len(), i + 1));
                full_text.push_str(page_text);
            }

            let mut chunks = chunk_string(&source, "unknown", &full_text, tokenizer, 200, 40);

            for chunk in chunks.iter_mut() {
                if let Some(pos) = full_text.find(chunk.text.as_str()) {
                    let page = page_boundaries
                        .iter()
                        .rev()
                        .find(|(start, _)| *start <= pos)
                        .map(|(_, page)| *page)
                        .unwrap_or(1);
                    chunk.unit = format!("page_{}", page);
                }
            }

            Ok(chunks)
        }
        Some("pptx") => {
            // load presentation
            let mut presentation = PresentationContainer::open(
                path,
                ParserConfig::builder().extract_images(false).build(),
            )?;
            let markdown = presentation.convert_to_md()?;

            let mut chunks = Vec::new();
            let mut current_unit = String::from("header");
            let mut buffer = String::new();

            for line in markdown.lines() {
                if let Some(rest) = line.strip_prefix("<!-- Slide ") {
                    if let Some(num) = rest.strip_suffix(" -->") {
                        if !buffer.trim().is_empty() {
                            chunks.extend(chunk_string(
                                &source,
                                &current_unit,
                                buffer.trim(),
                                &tokenizer,
                                200,
                                40,
                            ));
                        }
                        buffer.clear();
                        current_unit = format!("slide_{}", num);
                        continue;
                    }
                }
                buffer.push_str(line);
                buffer.push('\n');
            }
            if !buffer.trim().is_empty() {
                chunks.extend(chunk_string(
                    &source,
                    &current_unit,
                    buffer.trim(),
                    &tokenizer,
                    200,
                    40,
                ));
            }
            Ok(chunks)
        }
        Some("md") => {
            let contents = fs::read_to_string(path)?;

            let mut chunks = Vec::new();
            let mut current_unit = String::from("intro");
            let mut buffer = String::new();

            for line in contents.lines() {
                if line.starts_with("## ") {
                    if !buffer.trim().is_empty() {
                        chunks.extend(chunk_string(
                            &source,
                            &current_unit,
                            buffer.trim(),
                            &tokenizer,
                            200,
                            40,
                        ));
                    }
                    buffer.clear();
                    current_unit = line.trim_start_matches("## ").trim().to_string();
                    continue;
                }
                buffer.push_str(line);
                buffer.push('\n');
            }
            if !buffer.trim().is_empty() {
                chunks.extend(chunk_string(
                    &source,
                    &current_unit,
                    buffer.trim(),
                    &tokenizer,
                    200,
                    40,
                ));
            }
            Ok(chunks)
        }
        _ => Ok(vec![]),
    }
}
