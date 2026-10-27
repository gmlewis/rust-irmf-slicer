//! Client for the shared TSL -> WGSL compiler
//! (`irmf-examples/tools/tsl-compiler`).
//!
//! TSL is JavaScript, so the slicer cannot evaluate it in-process. Instead it
//! invokes `irmf-tsl-compile`, which emits a standalone WGSL function that the
//! existing WGSL renderer splices into its footer.

use crate::IrmfResult;
use crate::irmf::IrmfError;
use serde::Deserialize;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

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

fn compile_tsl_with_exe(exe: &str, source: &str, materials: usize) -> IrmfResult<String> {
    let mut cmd = Command::new(exe);
    cmd.arg("--json")
        .arg("--materials")
        .arg(materials.to_string())
        .arg("-");
    if let Some(dir) = cache_dir() {
        cmd.arg("--cache-dir").arg(dir);
    }
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

    let output = child.wait_with_output()?;
    if output.stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.contains("TSL compiler not") {
            return Err(IrmfError::TslCompilerUnavailable);
        }
        return Err(IrmfError::TslCompileError(if stderr.is_empty() {
            format!("compiler exited with {}", output.status)
        } else {
            stderr
        }));
    }

    let res: CompileResult = serde_json::from_slice(&output.stdout)?;
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
