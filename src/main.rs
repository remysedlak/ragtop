use std::error::Error;

use walkdir::WalkDir;

pub mod extract;
pub mod hf;

fn main() -> Result<(), Box<dyn Error>> {
    for entry in WalkDir::new("./test_folder") {
        let entry = entry?;
        if entry.file_type().is_file() {
            match extract::get_chunks_from_file(&entry) {
                Ok(text) => {
                    println!("{:#?}", text);
                }
                Err(e) => eprintln!("Failed on {}: {e}", entry.path().display()),
            }
        }
    }
    Ok(())
}
