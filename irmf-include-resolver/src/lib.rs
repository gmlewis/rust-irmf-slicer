use regex::Regex;
use std::collections::HashSet;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ResolverError {
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub async fn resolve_includes(source: &str) -> Result<String, ResolverError> {
    let mut resolved_source = String::new();
    let include_re = Regex::new(r#"^#include\s+"([^"]+)""#).unwrap();
    let mut visited = HashSet::new();

    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(caps) = include_re.captures(trimmed) {
            let inc = &caps[1];
            if !visited.contains(inc) {
                visited.insert(inc.to_string());
                if let Some(url) = parse_include_url(inc) {
                    let content = fetch_url(&url).await?;
                    resolved_source.push_str(&content);
                    resolved_source.push('\n');
                    continue;
                }
            }
        }
        resolved_source.push_str(line);
        resolved_source.push('\n');
    }

    Ok(resolved_source)
}

fn parse_include_url(inc: &str) -> Option<String> {
    if !inc.ends_with(".glsl") && !inc.ends_with(".wgsl") {
        return None;
    }

    const LYGIA_BASE_URL: &str = "https://lygia.xyz";
    const GITHUB_RAW_PREFIX: &str = "https://raw.githubusercontent.com/";

    if inc.starts_with("lygia.xyz/") {
        Some(format!("{}/{}", LYGIA_BASE_URL, &inc[10..]))
    } else if inc.starts_with("lygia/") {
        Some(format!("{}/{}", LYGIA_BASE_URL, &inc[6..]))
    } else if inc.starts_with("github.com/") {
        let location = inc[11..].replace("/blob/", "/");
        Some(format!("{}{}", GITHUB_RAW_PREFIX, location))
    } else {
        None
    }
}

async fn fetch_url(url: &str) -> Result<String, ResolverError> {
    println!("Fetching {}", url);
    let content = reqwest::get(url).await?.text().await?;
    Ok(content)
}
