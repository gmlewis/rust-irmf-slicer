//! Utility for resolving `#include` directives in IRMF shaders.
//!
//! This crate provides logic to fetch remote includes from `lygia.xyz`
//! and `github.com`, supporting both GLSL and WGSL shaders.
//!
//! Remote includes are fetched code. Callers must opt in explicitly (the
//! slicers only do so with `--remote-includes`), and must never resolve
//! includes in a TSL body, which is JavaScript.
//!
//! For more information about the IRMF format and its capabilities, visit the
//! [official IRMF website](https://irmf.io).

use regex::Regex;
use reqwest::redirect::Policy;
use reqwest::{Client, StatusCode, Url};
use std::time::Duration;
use thiserror::Error;

/// Maximum bytes accepted for a single include.
pub const MAX_INCLUDE_BYTES: usize = 1 << 20;
/// Maximum number of `#include` directives resolved per model.
pub const MAX_INCLUDES: usize = 16;
/// Timeout for each include request.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
/// Maximum number of HTTP redirects followed.
const MAX_REDIRECTS: usize = 5;

/// Error type for include resolution operations.
#[derive(Error, Debug)]
pub enum ResolverError {
    /// A network error occurred while fetching a remote include.
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    /// The server returned a non-success status.
    #[error("HTTP {0} fetching {1}")]
    Http(StatusCode, String),
    /// The include URL is not allowed.
    #[error("Disallowed include URL: {0}")]
    DisallowedUrl(String),
    /// The include is larger than the limit.
    #[error("Include {url} exceeds the {limit}-byte limit")]
    TooLarge { url: String, limit: usize },
    /// Too many includes were requested.
    #[error("Too many remote includes (limit is {0})")]
    TooManyIncludes(usize),
    /// The fetched include is not valid UTF-8.
    #[error("Include is not valid UTF-8: {0}")]
    InvalidUtf8(String),
}

/// Reports whether `host` is an allowed include host.
fn is_allowed_host(host: &str) -> bool {
    matches!(host, "lygia.xyz" | "raw.githubusercontent.com")
}

/// Returns true when `source` contains an `#include` directive line.
pub fn has_include_directive(source: &str) -> bool {
    source
        .lines()
        .any(|line| line.trim_start().starts_with("#include"))
}

fn http_client() -> Result<Client, ResolverError> {
    Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .redirect(Policy::custom(|attempt| {
            if attempt.previous().len() >= MAX_REDIRECTS {
                return attempt.error("too many redirects");
            }
            let url = attempt.url().clone();
            if url.scheme() == "https" && url.host_str().map(is_allowed_host).unwrap_or(false) {
                attempt.follow()
            } else {
                attempt.error(format!("redirect to disallowed include host: {}", url))
            }
        }))
        .user_agent("irmf-include-resolver/1.0 (+https://github.com/gmlewis/rust-irmf-slicer)")
        .build()
        .map_err(ResolverError::NetworkError)
}

/// Options for include resolution.
#[derive(Debug, Clone, Copy, Default)]
pub struct IncludeOptions {
    /// Accept GitHub includes that name a branch or tag instead of a 40-hex
    /// commit. Less reproducible: the included code can change after review.
    /// Off by default; only enable on an explicit operator request.
    pub allow_mutable: bool,
}

/// Resolves `#include` directives in the given shader source, requiring pinned
/// commits for GitHub URLs.
///
/// Currently supports:
/// - `lygia.xyz/` or `lygia/` prefix (maps to `https://lygia.xyz`)
/// - `github.com/` prefix (maps to `https://raw.githubusercontent.com`)
///
/// Only `.glsl` and `.wgsl` files are supported. A failed fetch is an error:
/// silently dropping the include would produce a shader that differs from the
/// author's intent.
pub async fn resolve_includes(source: &str) -> Result<String, ResolverError> {
    resolve_includes_with_options(source, IncludeOptions::default()).await
}

/// Resolves `#include` directives using explicit options.
pub async fn resolve_includes_with_options(
    source: &str,
    options: IncludeOptions,
) -> Result<String, ResolverError> {
    let client = http_client()?;
    let mut resolved_source = String::new();
    let include_re = Regex::new("^#include\\s+\"([^\"]+)\"").unwrap();
    let mut count = 0;

    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(caps) = include_re.captures(trimmed) {
            let inc = &caps[1];
            let url =
                parse_include_url_with_options(inc, options.allow_mutable).ok_or_else(|| {
                    let hint = if options.allow_mutable {
                        ""
                    } else {
                        " (pass --allow-mutable-includes to accept a branch or tag)"
                    };
                    ResolverError::DisallowedUrl(format!(
                        "unsupported or unpinned include: {:?}{hint}",
                        inc.escape_debug().to_string()
                    ))
                })?;
            if options.allow_mutable && parse_include_url(inc).is_none() {
                println!("WARNING: fetching mutable include: {:?}", url);
            }
            if count >= MAX_INCLUDES {
                return Err(ResolverError::TooManyIncludes(MAX_INCLUDES));
            }
            let content = fetch_url(&client, &url).await?;
            count += 1;
            resolved_source.push_str(&content);
            resolved_source.push('\n');
            continue;
        }
        resolved_source.push_str(line);
        resolved_source.push('\n');
    }

    Ok(resolved_source)
}

/// Parses an include path into a full URL, requiring a pinned commit.
fn parse_include_url(inc: &str) -> Option<String> {
    parse_include_url_with_options(inc, false)
}

fn parse_include_url_with_options(inc: &str, allow_mutable: bool) -> Option<String> {
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
        // Prefer an immutable commit hash. A branch or tag can be moved by
        // the repository owner, silently changing the code a model executes.
        if let Some(caps) = github_include_re().captures(stripped) {
            if !safe_ref_and_path(&caps[3], &caps[4]) {
                return None;
            }
            return Some(format!(
                "{}{}/{}/{}/{}",
                GITHUB_RAW_PREFIX, &caps[1], &caps[2], &caps[3], &caps[4]
            ));
        }
        if !allow_mutable {
            return None;
        }
        let caps = github_mutable_include_re().captures(stripped)?;
        if !safe_ref_and_path(&caps[3], &caps[4]) {
            return None;
        }
        Some(format!(
            "{}{}/{}/{}/{}",
            GITHUB_RAW_PREFIX, &caps[1], &caps[2], &caps[3], &caps[4]
        ))
    } else {
        None
    }
}

fn github_include_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^([^/]+)/([^/]+)/(?:(?:blob|raw)/)?([0-9a-fA-F]{40})/(.+)$").unwrap()
    })
}

fn github_mutable_include_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^([^/]+)/([^/]+)/(?:blob|raw)/([^/]+)/(.+)$").unwrap())
}

/// Rejects refs and paths that would change the constructed URL or traverse
/// the fetched path.
fn safe_ref_and_path(reference: &str, path: &str) -> bool {
    if reference.is_empty()
        || reference == "."
        || reference == ".."
        || reference.contains("..")
        || reference
            .chars()
            .any(|c| matches!(c, '\\' | '?' | '#' | '%'))
    {
        return false;
    }
    if path
        .chars()
        .any(|c| matches!(c, '\\' | '?' | '#' | '%') || c.is_control() || c.is_whitespace())
    {
        return false;
    }
    !path.split('/').any(|part| part == "." || part == "..")
}

/// Fetches the content of an allowlisted HTTPS URL with a size cap.
async fn fetch_url(client: &Client, url: &str) -> Result<String, ResolverError> {
    let parsed =
        Url::parse(url).map_err(|e| ResolverError::DisallowedUrl(format!("{url:?}: {e}")))?;
    if parsed.scheme() != "https"
        || !parsed.host_str().map(is_allowed_host).unwrap_or(false)
        || !parsed.username().is_empty()
    {
        return Err(ResolverError::DisallowedUrl(url.escape_debug().to_string()));
    }

    println!("Fetching {:?}", url);
    let mut resp = client.get(parsed).send().await?;
    if !resp.status().is_success() {
        return Err(ResolverError::Http(
            resp.status(),
            url.escape_debug().to_string(),
        ));
    }
    if resp
        .content_length()
        .is_some_and(|len| len > MAX_INCLUDE_BYTES as u64)
    {
        return Err(ResolverError::TooLarge {
            url: url.escape_debug().to_string(),
            limit: MAX_INCLUDE_BYTES,
        });
    }

    // Read incrementally so a hostile server cannot exceed the cap by
    // omitting Content-Length.
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await? {
        if body.len() + chunk.len() > MAX_INCLUDE_BYTES {
            return Err(ResolverError::TooLarge {
                url: url.escape_debug().to_string(),
                limit: MAX_INCLUDE_BYTES,
            });
        }
        body.extend_from_slice(&chunk);
    }

    String::from_utf8(body).map_err(|e| ResolverError::InvalidUtf8(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_include_prefixes_to_https_urls() {
        assert_eq!(
            parse_include_url("lygia/math/decimation.glsl").as_deref(),
            Some("https://lygia.xyz/math/decimation.glsl")
        );
        assert_eq!(
            parse_include_url("lygia.xyz/math/decimation.glsl").as_deref(),
            Some("https://lygia.xyz/math/decimation.glsl")
        );
        let sha = "6bb268f05fbe1c68a68ec1e66ce4c5d335bdf021";
        let want = format!("https://raw.githubusercontent.com/user/repo/{sha}/x.wgsl");
        assert_eq!(
            parse_include_url(&format!("github.com/user/repo/blob/{sha}/x.wgsl")).as_deref(),
            Some(want.as_str())
        );
    }

    #[test]
    fn rejects_other_paths_and_extensions() {
        assert!(parse_include_url("/etc/passwd").is_none());
        assert!(parse_include_url("lygia/x.txt").is_none());
        // Branches and tags are mutable, so they are not accepted by default.
        assert!(parse_include_url("github.com/user/repo/blob/main/x.wgsl").is_none());
        assert!(parse_include_url("github.com/user/repo/blob/v1.0.0/x.wgsl").is_none());
    }

    #[test]
    fn mutable_refs_require_an_explicit_opt_in() {
        assert_eq!(
            parse_include_url_with_options("github.com/user/repo/blob/main/x.wgsl", true)
                .as_deref(),
            Some("https://raw.githubusercontent.com/user/repo/main/x.wgsl")
        );
        // Traversal and URL-injection attempts stay rejected even with the
        // opt-in enabled.
        assert!(
            parse_include_url_with_options("github.com/user/repo/blob/main/../../x.wgsl", true)
                .is_none()
        );
        assert!(
            parse_include_url_with_options("github.com/user/repo/blob/main/x.wgsl?raw=1", true)
                .is_none()
        );
    }

    #[test]
    fn only_allows_known_https_hosts() {
        assert!(is_allowed_host("lygia.xyz"));
        assert!(is_allowed_host("raw.githubusercontent.com"));
        assert!(!is_allowed_host("evil.example"));
        assert!(!is_allowed_host("127.0.0.1"));
    }

    #[test]
    fn detects_include_directives() {
        assert!(has_include_directive("#include \"lygia/x.glsl\"\n"));
        assert!(has_include_directive("  #include \"lygia/x.glsl\"\n"));
        assert!(!has_include_directive("// #include \"lygia/x.glsl\"\n"));
        assert!(!has_include_directive("void mainModel4() {}\n"));
    }
}
