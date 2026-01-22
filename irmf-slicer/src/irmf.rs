//! IRMF model and parsing logic.

use base64::{Engine, prelude::BASE64_STANDARD};
use flate2::read::GzDecoder;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::io::Read;
use thiserror::Error;

/// Error type for IRMF operations.
#[derive(Error, Debug)]
pub enum IrmfError {
    /// The leading '/*{' comment was not found.
    #[error("Unable to find leading '/*{{\'")]
    MissingLeadingComment,
    /// The trailing '}*/' comment was not found.
    #[error("Unable to find trailing '}}*/'")]
    MissingTrailingComment,
    /// An error occurred during JSON parsing.
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),
    /// An error occurred during Base64 decoding.
    #[error("Base64 decoding error: {0}")]
    Base64Error(#[from] base64::DecodeError),
    /// An IO error occurred.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    /// The IRMF version is not supported.
    #[error("Unsupported IRMF version: {0}")]
    UnsupportedVersion(String),
    /// The shader language is not supported.
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),
    /// The material list is invalid.
    #[error("Invalid materials: {0}")]
    InvalidMaterials(String),
    /// The Maximum Bounding Box (MBB) is invalid.
    #[error("Invalid MBB: {0}")]
    InvalidMbb(String),
    /// The encoding is not supported.
    #[error("Unsupported encoding: {0}")]
    UnsupportedEncoding(String),

    /// A general renderer error occurred.
    #[error("Renderer error: {0}")]
    RendererError(String),
    /// Failed to find a suitable WGPU adapter.
    #[error("WGPU adapter error")]
    WgpuAdapterError,
    /// Failed to request a WGPU device.
    #[error("WGPU device error: {0}")]
    WgpuDeviceError(#[from] wgpu::RequestDeviceError),
    /// An error occurred with a WGPU buffer.
    #[error("WGPU buffer error: {0}")]
    WgpuBufferError(#[from] wgpu::BufferAsyncError),
    /// Failed to receive data from a channel.
    #[error("Receive error: {0}")]
    RecvError(#[from] std::sync::mpsc::RecvError),
    /// An error occurred during shader compilation.
    #[error("Shader compilation error: {0}")]
    ShaderError(String),
}

/// The header of an IRMF model.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IrmfHeader {
    /// Author of the model.
    pub author: Option<String>,
    /// License of the model.
    pub license: Option<String>,
    /// Date the model was created.
    pub date: Option<String>,
    /// Encoding of the shader (e.g., "gzip", "gzip+base64").
    pub encoding: Option<String>,
    /// IRMF version.
    #[serde(rename = "irmf")]
    pub irmf_version: String,
    /// GLSL version if applicable.
    pub glsl_version: Option<String>,
    /// Language of the shader ("glsl" or "wgsl").
    pub language: Option<String>, // Default is "glsl"
    /// List of material names.
    pub materials: Vec<String>,
    /// Maximum coordinates of the bounding box.
    pub max: [f32; 3],
    /// Minimum coordinates of the bounding box.
    pub min: [f32; 3],
    /// Additional notes.
    pub notes: Option<String>,
    /// Format-specific options.
    pub options: Option<serde_json::Value>,
    /// Title of the model.
    pub title: Option<String>,
    /// Units of measurement (e.g., "mm").
    pub units: String,
    /// Version of the model.
    pub version: Option<String>,
}

/// A parsed IRMF model.
#[derive(Debug)]
pub struct IrmfModel {
    /// The model's header.
    pub header: IrmfHeader,
    /// The decompressed/decoded shader source.
    pub shader: String,
}

impl IrmfModel {
    /// Parses an IRMF model from a byte slice.
    ///
    /// The input data should contain the JSON header in a leading comment,
    /// followed by the shader payload.
    pub fn new(data: &[u8]) -> Result<Self, IrmfError> {
        let (header, shader_payload) = parse_header(data)?;
        let shader = decode_shader(&header, shader_payload)?;

        let model = IrmfModel { header, shader };
        model.validate()?;

        Ok(model)
    }

    /// Validates the IRMF model according to the specification.
    fn validate(&self) -> Result<(), IrmfError> {
        if self.header.irmf_version != "1.0" {
            return Err(IrmfError::UnsupportedVersion(
                self.header.irmf_version.clone(),
            ));
        }
        if self.header.materials.is_empty() {
            return Err(IrmfError::InvalidMaterials(
                "Must list at least one material name".into(),
            ));
        }
        if self.header.materials.len() > 16 {
            return Err(IrmfError::InvalidMaterials(format!(
                "IRMF 1.0 only supports up to 16 materials, found {}",
                self.header.materials.len()
            )));
        }

        for i in 0..3 {
            if self.header.min[i] >= self.header.max[i] {
                return Err(IrmfError::InvalidMbb(format!(
                    "min[{}] ({}) must be strictly less than max[{}] ({})",
                    i, self.header.min[i], i, self.header.max[i]
                )));
            }
        }

        let lang = self.header.language.as_deref().unwrap_or("glsl");
        match lang {
            "glsl" | "" | "wgsl" => {} // No-op
            _ => return Err(IrmfError::UnsupportedLanguage(lang.into())),
        }

        Ok(())
    }
}

/// Parses the JSON header from the IRMF data.
fn parse_header(data: &[u8]) -> Result<(IrmfHeader, &[u8]), IrmfError> {
    let start_tag = b"/*{";
    let end_tag = b"}*/";

    if !data.starts_with(start_tag) {
        return Err(IrmfError::MissingLeadingComment);
    }

    let end_index = data
        .windows(end_tag.len())
        .position(|window| window == end_tag)
        .ok_or(IrmfError::MissingTrailingComment)?;

    let json_str = String::from_utf8_lossy(&data[2..end_index + 1]);

    // Simple fix for trailing commas and unquoted keys
    let mut cleaned_json = json_str.into_owned();

    // Remove trailing commas before closing brace or bracket
    let re_comma = Regex::new(r",\s*([\}\]])").unwrap();
    cleaned_json = re_comma.replace_all(&cleaned_json, "$1").into_owned();

    let keys = [
        "author",
        "license",
        "date",
        "encoding",
        "irmf",
        "glslVersion",
        "language",
        "materials",
        "max",
        "min",
        "notes",
        "options",
        "title",
        "units",
        "version",
    ];

    for key in keys {
        // Look for unquoted keys: must be preceded by { or , or whitespace, followed by :
        let re_key = Regex::new(&format!(r"([{{,\s]){}\s*:", key)).unwrap();
        cleaned_json = re_key
            .replace_all(&cleaned_json, |caps: &regex::Captures| {
                format!("{}\"{}\":", &caps[1], key)
            })
            .into_owned();
    }

    let header: IrmfHeader = serde_json::from_str(&cleaned_json)?;

    let payload = &data[end_index + end_tag.len()..];
    let mut start = 0;
    while start < payload.len() && (payload[start] as char).is_whitespace() {
        start += 1;
    }

    Ok((header, &payload[start..]))
}

/// Decodes the shader payload based on the encoding specified in the header.
fn decode_shader(header: &IrmfHeader, payload: &[u8]) -> Result<String, IrmfError> {
    match header.encoding.as_deref() {
        Some("gzip") => {
            let mut decoder = GzDecoder::new(payload);
            let mut shader = String::new();
            decoder
                .read_to_string(&mut shader)
                .map_err(IrmfError::IoError)?;
            Ok(shader)
        }
        Some("gzip+base64") => {
            // Remove all whitespace from base64 payload
            let payload_str = std::str::from_utf8(payload).unwrap_or("");
            let cleaned_payload: String =
                payload_str.chars().filter(|c| !c.is_whitespace()).collect();
            let decoded = BASE64_STANDARD.decode(&cleaned_payload)?;
            let mut decoder = GzDecoder::new(&decoded[..]);
            let mut shader = String::new();
            decoder
                .read_to_string(&mut shader)
                .map_err(IrmfError::IoError)?;
            Ok(shader)
        }
        None | Some("") => Ok(String::from_utf8_lossy(payload).into_owned()),
        Some(enc) => Err(IrmfError::UnsupportedEncoding(enc.into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sphere() {
        let data = b"/*{
  \"author\": \"Glenn M. Lewis\",
  \"license\": \"Apache-2.0\",
  \"date\": \"2019-06-30\",
  \"irmf\": \"1.0\",
  \"language\": \"glsl\",
  \"materials\": [\"AISI 1018 steel\"],
  \"max\": [5,5,5],
  \"min\": [-5,-5,-5],
  \"notes\": \"Simple IRMF shader - Hello, Sphere!\",
  \"options\": {},
  \"title\": \"10mm diameter Sphere\",
  \"units\": \"mm\",
  \"version\": \"1.0\"
}*/

void mainModel4(out vec4 materials, in vec3 xyz) {
  const float radius = 5.0;
  float r = length(xyz);
  materials[0] = r <= radius ? 1.0 : 0.0;
}";
        let model = IrmfModel::new(data).unwrap();
        assert_eq!(model.header.irmf_version, "1.0");
        assert_eq!(model.header.materials[0], "AISI 1018 steel");
        assert_eq!(model.header.units, "mm");
        assert!(model.shader.contains("void mainModel4"));
    }

    #[test]
    fn test_parse_sphere_wgsl() {
        let data = b"/*{
  author: \"Glenn M. Lewis\",
  license: \"Apache-2.0\",
  date: \"2019-06-30\",
  irmf: \"1.0\",
  language: \"wgsl\",
  materials: [\"AISI 1018 steel\"],
  max: [5,5,5],
  min: [-5,-5,-5],
  notes: \"Simple IRMF shader - Hello, Sphere!\",
  options: {},
  title: \"10mm diameter Sphere\",
  units: \"mm\",
  version: \"1.0\",
}*/

fn mainModel4(xyz: vec3f) -> vec4f {
  let radius = 5.0;
  let r = length(xyz);
  var materials = vec4f(0.0);
  materials[0] = select(0.0, 1.0, r <= radius);
  return materials;
}";
        let model = IrmfModel::new(data).unwrap();
        assert_eq!(model.header.irmf_version, "1.0");
        assert_eq!(model.header.language.as_deref(), Some("wgsl"));
        assert!(model.shader.contains("fn mainModel4"));
    }
}
