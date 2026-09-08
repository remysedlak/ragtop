use pptx_to_md::{ParserConfig, PresentationContainer};
use std::{error::Error, fs};
use walkdir::DirEntry;

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
    chunk_words: usize,
    overlap_words: usize,
) -> Vec<Chunk> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    loop {
        let end = (start + chunk_words).min(words.len());
        chunks.push(Chunk {
            source: source.to_string(),
            unit: unit.to_string(),
            text: words[start..end].join(" "),
        });

        if end == words.len() {
            break;
        }
        start = end - overlap_words;
    }

    chunks
}

/// Extracts a file's text, splits it into structural units (page/slide/
/// header/paragraph depending on file type), and chunks each unit.
pub fn get_chunks_from_file(entry: &DirEntry) -> Result<Vec<Chunk>, Box<dyn Error>> {
    let path = entry.path();
    let ext = path.extension().and_then(|e| e.to_str());
    let source = path.file_name().unwrap().to_string_lossy().to_string();

    match ext {
        Some("txt") => {
            println!("TEXT:{}", path.display());
            let contents = fs::read_to_string(path)?;
            let mut chunks = Vec::new();
            for (i, para) in contents
                .split("\n\n")
                .filter(|p| !p.trim().is_empty())
                .enumerate()
            {
                let unit = format!("para_{}", i + 1);
                chunks.extend(chunk_string(&source, &unit, para.trim(), 200, 40));
            }
            Ok(chunks)
        }
        Some("pdf") => {
            println!("PDF:{}", path.display());
            let bytes = fs::read(path)?;
            let pages = pdf_extract::extract_text_from_mem_by_pages(&bytes)?;
            let mut chunks = Vec::new();
            for (i, page_text) in pages.into_iter().enumerate() {
                if page_text.trim().is_empty() {
                    continue;
                }
                let unit = format!("page_{}", i + 1);
                chunks.extend(chunk_string(&source, &unit, &page_text, 200, 40));
            }
            Ok(chunks)
        }
        Some("pptx") => {
            println!("PPTX:{}", path.display());
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
                chunks.extend(chunk_string(&source, &current_unit, buffer.trim(), 200, 40));
            }
            Ok(chunks)
        }
        Some("md") => {
            println!("MD:{}", path.display());
            let contents = fs::read_to_string(path)?;

            let mut chunks = Vec::new();
            let mut current_unit = String::from("intro");
            let mut buffer = String::new();

            for line in contents.lines() {
                if line.starts_with("## ") {
                    if !buffer.trim().is_empty() {
                        chunks.extend(chunk_string(&source, &current_unit, buffer.trim(), 200, 40));
                    }
                    buffer.clear();
                    current_unit = line.trim_start_matches("## ").trim().to_string();
                    continue;
                }
                buffer.push_str(line);
                buffer.push('\n');
            }
            if !buffer.trim().is_empty() {
                chunks.extend(chunk_string(&source, &current_unit, buffer.trim(), 200, 40));
            }
            Ok(chunks)
        }
        _ => Ok(vec![]),
    }
}
