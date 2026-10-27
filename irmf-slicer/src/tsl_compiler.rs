//! Client for the shared TSL -> WGSL compiler
//! (`irmf-examples/tools/tsl-compiler`).
//!
//! TSL is JavaScript, so the slicer cannot evaluate it in-process. Instead it
//! invokes `irmf-tsl-compile`, which emits a standalone WGSL function that the
//! existing WGSL renderer splices into its footer.

use crate::IrmfResult;
use crate::irmf::IrmfError;
use serde::Deserialize;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

/// Hard limit for one TSL compilation.
const COMPILE_TIMEOUT: Duration = Duration::from_secs(60);
/// Maximum JSON output accepted from the compiler.
const MAX_STDOUT_BYTES: usize = 16 << 20;
/// Maximum stderr captured for diagnostics.
const MAX_STDERR_BYTES: usize = 64 << 10;

static TSL_EXE: OnceLock<String> = OnceLock::new();
static TSL_CACHE: OnceLock<String> = OnceLock::new();

/// Overrides the compiler executable (otherwise $IRMF_TSL_COMPILER or PATH).
/// Must be called before the first TSL model is rendered.
pub fn set_tsl_compiler(exe: impl Into<String>) {
    let _ = TSL_EXE.set(exe.into());
}

/// Overrides the compilation cache directory (otherwise $IRMF_TSL_CACHE_DIR).
pub fn set_tsl_cache_dir(dir: impl Into<String>) {
    let _ = TSL_CACHE.set(dir.into());
}

fn cache_dir() -> Option<String> {
    TSL_CACHE
        .get()
        .cloned()
        .or_else(|| std::env::var("IRMF_TSL_CACHE_DIR").ok())
        .filter(|s| !s.is_empty())
}

#[derive(Debug, Deserialize)]
struct Diagnostic {
    #[serde(default)]
    line: u32,
    #[serde(default)]
    col: u32,
    message: String,
}

#[derive(Debug, Deserialize)]
struct CompileResult {
    ok: bool,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    diagnostics: Vec<Diagnostic>,
}

/// Compiles a TSL body into a standalone WGSL function.
pub fn compile_tsl(source: &str, materials: usize) -> IrmfResult<String> {
    let exe = TSL_EXE
        .get()
        .cloned()
        .or_else(|| std::env::var("IRMF_TSL_COMPILER").ok())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "irmf-tsl-compile".to_string());
    compile_tsl_with_exe(&exe, source, materials)
}

fn read_capped<R: Read>(reader: R, limit: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    reader.take(limit as u64 + 1).read_to_end(&mut buf)?;
    Ok(buf)
}

fn compile_tsl_with_exe(exe: &str, source: &str, materials: usize) -> IrmfResult<String> {
    let mut cmd = Command::new(exe);
    cmd.arg("--json")
        .arg("--materials")
        .arg(materials.to_string())
        .arg("-");
    if let Some(dir) = cache_dir() {
        cmd.arg("--cache-dir").arg(dir);
    }
    // The TSL body is untrusted code. Do not leak the parent's environment
    // (tokens, credentials) to the compiler, and do not run it in the model's
    // directory. The compiler additionally runs the body under Node's
    // permission model (no network, no writes, read-only package access).
    cmd.env_clear();
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", path);
    }
    if let Ok(tmp) = std::env::var("TMPDIR") {
        cmd.env("TMPDIR", tmp);
    }
    cmd.env("NO_COLOR", "1");
    cmd.current_dir(std::env::temp_dir());
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => IrmfError::TslCompilerUnavailable,
        _ => IrmfError::IoError(e),
    })?;

    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| IrmfError::TslCompileError("cannot write to compiler stdin".into()))?;
        stdin.write_all(source.as_bytes())?;
    }

    // Drain the pipes on separate threads so a verbose compiler cannot block.
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| IrmfError::TslCompileError("cannot read compiler stdout".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| IrmfError::TslCompileError("cannot read compiler stderr".into()))?;
    let out_thread = thread::spawn(move || read_capped(stdout, MAX_STDOUT_BYTES));
    let err_thread = thread::spawn(move || read_capped(stderr, MAX_STDERR_BYTES));

    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        match child.try_wait()? {
            Some(status) => break Some(status),
            None if start.elapsed() >= COMPILE_TIMEOUT => {
                let _ = child.kill();
                let _ = child.wait();
                timed_out = true;
                break None;
            }
            None => thread::sleep(Duration::from_millis(25)),
        }
    };

    let stdout = out_thread
        .join()
        .map_err(|_| IrmfError::TslCompileError("compiler stdout reader panicked".into()))??;
    let stderr = err_thread
        .join()
        .map_err(|_| IrmfError::TslCompileError("compiler stderr reader panicked".into()))??;

    if timed_out {
        return Err(IrmfError::TslCompileError(format!(
            "TSL compile timed out after {:?}",
            COMPILE_TIMEOUT
        )));
    }
    if stdout.len() > MAX_STDOUT_BYTES {
        return Err(IrmfError::ResourceLimit(format!(
            "TSL compiler output exceeds {} bytes",
            MAX_STDOUT_BYTES
        )));
    }
    if stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&stderr).trim().to_string();
        if stderr.contains("TSL compiler not") {
            return Err(IrmfError::TslCompilerUnavailable);
        }
        return Err(IrmfError::TslCompileError(if stderr.is_empty() {
            format!(
                "compiler exited with {}",
                status.map(|s| s.to_string()).unwrap_or_default()
            )
        } else {
            stderr
        }));
    }

    let res: CompileResult = serde_json::from_slice(&stdout)?;
    if !res.ok {
        if let Some(d) = res.diagnostics.first() {
            return Err(IrmfError::TslCompileError(format!(
                "line {}:{}: {}",
                d.line, d.col, d.message
            )));
        }
        return Err(IrmfError::TslCompileError("unknown compiler error".into()));
    }

    res.source
        .filter(|s| !s.is_empty())
        .ok_or_else(|| IrmfError::TslCompileError("compiler returned an empty shader".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_compiler_is_reported() {
        let err = compile_tsl_with_exe(
            "/nonexistent/irmf-tsl-compile",
            "const mainModel4 = (xyz) => [float(1)];",
            1,
        )
        .unwrap_err();
        assert!(matches!(err, IrmfError::TslCompilerUnavailable));
    }
}
