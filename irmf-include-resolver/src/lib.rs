//! Utility for resolving `#include` directives in IRMF shaders.
//!
//! This crate provides logic to fetch remote includes from `lygia.xyz`
//! and `github.com`, supporting both GLSL and WGSL shaders.
//!
//! For more information about the IRMF format and its capabilities, visit the
//! [official IRMF website](https://irmf.io).

use regex::Regex;
use thiserror::Error;

/// Error type for include resolution operations.
#[derive(Error, Debug)]
pub enum ResolverError {
    /// A network error occurred while fetching a remote include.
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
}

/// Resolves `#include` directives in the given shader source.
///
/// Currently supports:
/// - `lygia.xyz/` or `lygia/` prefix (maps to `https://lygia.xyz`)
/// - `github.com/` prefix (maps to `https://raw.githubusercontent.com`)
///
/// Only `.glsl` and `.wgsl` files are supported.
pub async fn resolve_includes(source: &str) -> Result<String, ResolverError> {
    let mut resolved_source = String::new();
    let include_re = Regex::new("^#include\\s+\"([^\"]+)\"").unwrap();

    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(caps) = include_re.captures(trimmed) {
            let inc = &caps[1];
            if let Some(url) = parse_include_url(inc) {
                if let Ok(content) = fetch_url(&url).await {
                    resolved_source.push_str(&content);
                    resolved_source.push('\n');
                }
                continue;
            }
        }
        resolved_source.push_str(line);
        resolved_source.push('\n');
    }

    Ok(resolved_source)
}

/// Parses an include path into a full URL.
fn parse_include_url(inc: &str) -> Option<String> {
    if !inc.ends_with(".glsl") && !inc.ends_with(".wgsl") {
        return None;
    }

    const LYGIA_BASE_URL: &str = "https://lygia.xyz";
    const GITHUB_RAW_PREFIX: &str = "https://raw.githubusercontent.com/";

    if let Some(stripped) = inc.strip_prefix("lygia.xyz/") {
        Some(format!("{}/{}", LYGIA_BASE_URL, stripped))
    } else if let Some(stripped) = inc.strip_prefix("lygia/") {
        Some(format!("{}/{}", LYGIA_BASE_URL, stripped))
    } else if let Some(stripped) = inc.strip_prefix("github.com/") {
        let location = stripped.replace("/blob/", "/");
        Some(format!("{}{}", GITHUB_RAW_PREFIX, location))
    } else {
        None
    }
}

/// Fetches the content of a URL.
async fn fetch_url(url: &str) -> Result<String, ResolverError> {
    println!("Fetching {}", url);
    let content = reqwest::get(url).await?.text().await?;
    Ok(content)
}
