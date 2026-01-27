//! Compresses IRMF shader payloads using gzip and optional base64 encoding.
//!
//! This tool reads an IRMF file, compresses the shader source code,
//! and updates the header to reflect the compression encoding.

use anyhow::{Context, Result};
use base64::{Engine, engine::general_purpose::STANDARD_NO_PAD as BASE64};
use clap::Parser;
use flate2::Compression;
use flate2::write::GzEncoder;
use irmf_slicer::IrmfModel;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input IRMF file
    input: PathBuf,

    /// Output IRMF file (default: <input>.compressed.irmf)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Use base64 encoding in addition to gzip
    #[arg(long)]
    base64: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let data = std::fs::read(&cli.input)
        .with_context(|| format!("Failed to read input file: {}", cli.input.display()))?;

    let model = IrmfModel::new(&data)
        .with_context(|| format!("Failed to parse IRMF model: {}", cli.input.display()))?;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(model.shader.as_bytes())?;
    let compressed_data = encoder.finish()?;

    let (encoding, payload) = if cli.base64 {
        let b64 = BASE64.encode(&compressed_data);
        ("gzip+base64", b64.into_bytes())
    } else {
        ("gzip", compressed_data)
    };

    let mut new_header = model.header;
    new_header.encoding = Some(encoding.to_string());

    let header_json = new_header.serialize_to_string();

    let output_path = cli.output.unwrap_or_else(|| {
        let mut path = cli.input.clone();
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("irmf");
        path.set_extension(format!("compressed.{}", extension));
        path
    });

    let mut output_file = File::create(&output_path)
        .with_context(|| format!("Failed to create output file: {}", output_path.display()))?;

    // Wrapping pretty JSON in /* and */ ensures /*{ and }*/ are on their own lines
    // because to_string_pretty puts { and } on their own lines.
    writeln!(output_file, "/*{}*/", header_json)?;
    output_file.write_all(&payload)?;

    println!(
        "Successfully wrote compressed IRMF to {}",
        output_path.display()
    );

    // Validate our own output
    let output_data = std::fs::read(&output_path)?;
    IrmfModel::validate_spec_compliance(&output_data)
        .context("Generated file is not spec-compliant")?;

    Ok(())
}
