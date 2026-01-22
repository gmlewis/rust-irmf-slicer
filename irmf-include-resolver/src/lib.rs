use regex::Regex;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ResolverError {
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
}

pub async fn resolve_includes(source: &str) -> Result<String, ResolverError> {
    let mut resolved_source = String::new();
    let include_re = Regex::new(r"^#include\s+\"([^\"]+)\\"" ).unwrap();

    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(caps) = include_re.captures(trimmed) {
            let inc = &caps[1];
            if let Some(url) = parse_include_url(inc) {
                if let Ok(content) = fetch_url(&url).await {
                    resolved_source.push_str(&content);
                    resolved_source.push('\n');
                }
                // Go implementation continues here, effectively dropping the line
                // if it was recognized as a URL (whether fetch succeeded or failed).
                continue;
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