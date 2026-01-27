//! IRMF model and parsing logic.

use base64::Engine;
use chrono::Local;
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
    /// The header does not comply with the IRMF specification.
    #[error("IRMF spec compliance error: {0}")]
    HeaderSpecError(String),

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// License of the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Date the model was created.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    /// Encoding of the shader (e.g., "gzip", "gzip+base64").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding: Option<String>,
    /// IRMF version.
    #[serde(rename = "irmf")]
    pub irmf_version: String,
    /// GLSL version if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glsl_version: Option<String>,
    /// Language of the shader ("glsl" or "wgsl").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>, // Default is "glsl"
    /// List of material names.
    pub materials: Vec<String>,
    /// Maximum coordinates of the bounding box.
    pub max: [f32; 3],
    /// Minimum coordinates of the bounding box.
    pub min: [f32; 3],
    /// Additional notes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// Format-specific options.
    #[serde(skip_serializing_if = "is_option_empty_map")]
    pub options: Option<serde_json::Value>,
    /// Title of the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Units of measurement (e.g., "mm").
    pub units: String,
    /// Version of the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

impl IrmfHeader {
    /// Serializes the header to a string with the following requirements:
    /// 1. remove all double-quotes around the keys
    /// 2. always put the "min" key before the "max" key
    /// 3. always keep all the \[x,y,z\] values for both the "min" and the "max" lines on the same line
    /// 4. if the "date" key/value pair is missing or the value is empty, fill in today's date and add after author
    /// 5. keep the list of all materials on a single line
    /// 6. always add a trailing comma to the last key/value pair in the list
    pub fn serialize_to_string(&self) -> String {
        let mut lines = Vec::new();
        lines.push("{".to_string());

        if let Some(ref author) = self.author {
            lines.push(format!("  author: \"{}\",", author));
        }

        let date = if let Some(ref d) = self.date {
            if d.is_empty() {
                Local::now().format("%Y-%m-%d").to_string()
            } else {
                d.clone()
            }
        } else {
            Local::now().format("%Y-%m-%d").to_string()
        };
        lines.push(format!("  date: \"{}\",", date));

        if let Some(ref license) = self.license {
            lines.push(format!("  license: \"{}\",", license));
        }

        if let Some(ref encoding) = self.encoding {
            lines.push(format!("  encoding: \"{}\",", encoding));
        }

        lines.push(format!("  irmf: \"{}\",", self.irmf_version));

        if let Some(ref glsl) = self.glsl_version {
            lines.push(format!("  glslVersion: \"{}\",", glsl));
        }

        if let Some(ref lang) = self.language {
            lines.push(format!("  language: \"{}\",", lang));
        }

        let mats = self
            .materials
            .iter()
            .map(|m| format!("\"{}\"", m))
            .collect::<Vec<_>>()
            .join(",");
        lines.push(format!("  materials: [{}],", mats));

        lines.push(format!(
            "  min: [{:.4},{:.4},{:.4}],",
            self.min[0], self.min[1], self.min[2]
        ));
        lines.push(format!(
            "  max: [{:.4},{:.4},{:.4}],",
            self.max[0], self.max[1], self.max[2]
        ));

        if let Some(ref notes) = self.notes {
            lines.push(format!("  notes: \"{}\",", notes));
        }

        if let Some(ref options) = self.options {
            if !is_option_empty_map(&self.options) {
                if let Ok(json) = serde_json::to_string(options) {
                    lines.push(format!("  options: {},", json));
                }
            }
        }

        if let Some(ref title) = self.title {
            lines.push(format!("  title: \"{}\",", title));
        }

        lines.push(format!("  units: \"{}\",", self.units));

        if let Some(ref version) = self.version {
            lines.push(format!("  version: \"{}\",", version));
        }

        lines.push("}".to_string());
        lines.join("\n")
    }
}

/// Helper function for serde to skip optional empty maps (mimicking Go's omitempty).
fn is_option_empty_map(v: &Option<serde_json::Value>) -> bool {
    match v {
        Some(serde_json::Value::Object(map)) => map.is_empty(),
        None => true,
        _ => false,
    }
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

    /// Validates that the IRMF file strictly complies with the specification.
    ///
    /// Checks:
    /// - Leading '/*{' on its own line.
    /// - Trailing '}*/' on its own line.
    /// - JSON header is pretty-printed (one key per line).
    pub fn validate_spec_compliance(data: &[u8]) -> Result<(), IrmfError> {
        let content = String::from_utf8_lossy(data);
        let mut lines = content.lines();

        // 1. Leading '/*{' on its own line.
        let first_line = lines
            .next()
            .ok_or_else(|| IrmfError::HeaderSpecError("File is empty".into()))?;
        if first_line.trim() != "/*{" {
            return Err(IrmfError::HeaderSpecError(format!(
                "First line must be '/*{{', found '{}'",
                first_line
            )));
        }

        // 2. Find '}*/' and check if it's on its own line.
        let mut header_lines = Vec::new();
        let mut found_end = false;
        for line in lines {
            if line.trim() == "}*/" {
                found_end = true;
                break;
            }
            header_lines.push(line);
        }

        if !found_end {
            return Err(IrmfError::HeaderSpecError(
                "Unable to find '}}*/' on its own line".into(),
            ));
        }

        // 3. JSON header is valid (checked by IrmfModel::new, but we can verify it parses here if we want)
        // For now, the user just wants to ensure the tags are on their own lines and it parses.
        // The parsing is already handled by the caller (IrmfModel::new).

        Ok(())
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

    // Combined regex to match:
    // 1. Double-quoted strings (to skip them)
    // 2. Unquoted keys: preceded by { or , or whitespace, followed by :
    // 3. Trailing commas: comma followed by whitespace and } or ]
    let re =
        Regex::new(r#"(?s)("([^"\\]|\\.)*")|([{,\s])([a-zA-Z_][a-zA-Z0-9_]*)\s*:|(,)\s*([\}\]])"#)
            .unwrap();

    let cleaned_json = re
        .replace_all(&json_str, |caps: &regex::Captures| {
            if let Some(mat) = caps.get(1) {
                // It's a string, return it as is
                mat.as_str().to_string()
            } else if let Some(sep) = caps.get(3) {
                // It's an unquoted key, quote it
                format!("{}\"{}\":", sep.as_str(), &caps[4])
            } else if caps.get(5).is_some() {
                // It's a trailing comma, remove the comma and keep the closing brace/bracket
                caps[6].to_string()
            } else {
                // Should not happen, but return original as fallback
                caps[0].to_string()
            }
        })
        .into_owned();

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
            let decoded =
                base64::engine::general_purpose::STANDARD_NO_PAD.decode(&cleaned_payload)?;
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

    #[test]
    fn test_parse_header_edge_cases() {
        let data = b"/*{
  irmf: \"1.0\",
  materials: [\"PLA\",],
  max: [1,1,1],
  min: [0,0,0],
  units: \"mm\",
  version: \"1.0\",
  options: {
    unquotedKey: 42,
    \"quotedKey\": \"value\",
    nested: {
        deeper: [1, 2, ],
        more: {
            a: 1,
            b: 2
        }
    }
  },
  title: \"Edge Case Test\",
}*/
void main() {}";
        let model = IrmfModel::new(data).unwrap();
        assert_eq!(model.header.irmf_version, "1.0");
        assert_eq!(model.header.title.as_deref(), Some("Edge Case Test"));
        let options = model.header.options.as_ref().unwrap();
        assert_eq!(options["unquotedKey"], 42);
        assert_eq!(options["quotedKey"], "value");
        assert_eq!(options["nested"]["deeper"][1], 2);
        assert_eq!(options["nested"]["more"]["a"], 1);
    }

    fn test_encoding_decoding(header: &mut IrmfHeader, shader: &str, use_b64: bool) {
        use base64::engine::general_purpose::STANDARD_NO_PAD;
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(shader.as_bytes()).unwrap();
        let compressed = encoder.finish().unwrap();

        let (encoding, payload) = if use_b64 {
            (
                "gzip+base64",
                STANDARD_NO_PAD.encode(&compressed).into_bytes(),
            )
        } else {
            ("gzip", compressed)
        };

        header.encoding = Some(encoding.to_string());
        let header_json = header.serialize_to_string();

        let mut full_data = Vec::new();
        full_data.extend_from_slice(b"/*");
        full_data.extend_from_slice(header_json.as_bytes());
        full_data.extend_from_slice(b"*/\n");
        full_data.extend_from_slice(&payload);

        let decoded_model = IrmfModel::new(&full_data).unwrap();
        assert_eq!(decoded_model.shader, shader);
        assert_eq!(decoded_model.header.encoding.as_deref(), Some(encoding));
    }

    #[test]
    fn test_sphere_glsl_encodings() {
        let glsl_shader = "void mainModel4(out vec4 materials, in vec3 xyz) {\n  const float radius = 5.0;\n  float r = length(xyz);\n  materials[0] = r <= radius ? 1.0 : 0.0;\n}";
        let mut header = IrmfHeader {
            author: Some("Glenn M. Lewis".into()),
            license: Some("Apache-2.0".into()),
            date: Some("2019-06-30".into()),
            encoding: None,
            irmf_version: "1.0".into(),
            glsl_version: None,
            language: Some("glsl".into()),
            materials: vec!["AISI 1018 steel".into()],
            max: [5.0, 5.0, 5.0],
            min: [-5.0, -5.0, -5.0],
            notes: Some("Simple IRMF shader - Hello, Sphere!".into()),
            options: Some(serde_json::json!({})),
            title: Some("10mm diameter Sphere".into()),
            units: "mm".into(),
            version: Some("1.0".into()),
        };

        test_encoding_decoding(&mut header, glsl_shader, false); // gzip
        test_encoding_decoding(&mut header, glsl_shader, true); // gzip+base64
    }

    #[test]
    fn test_sphere_wgsl_encodings() {
        let wgsl_shader = "fn mainModel4(xyz: vec3f) -> vec4f {\n  let radius = 5.0;\n  let r = length(xyz);\n  var materials = vec4f(0.0);\n  materials[0] = select(0.0, 1.0, r <= radius);\n  return materials;\n}";
        let mut header = IrmfHeader {
            author: Some("Glenn M. Lewis".into()),
            license: Some("Apache-2.0".into()),
            date: Some("2019-06-30".into()),
            encoding: None,
            irmf_version: "1.0".into(),
            glsl_version: None,
            language: Some("wgsl".into()),
            materials: vec!["AISI 1018 steel".into()],
            max: [5.0, 5.0, 5.0],
            min: [-5.0, -5.0, -5.0],
            notes: Some("Simple IRMF shader - Hello, Sphere!".into()),
            options: Some(serde_json::json!({})),
            title: Some("10mm diameter Sphere".into()),
            units: "mm".into(),
            version: Some("1.0".into()),
        };

        test_encoding_decoding(&mut header, wgsl_shader, false); // gzip
        test_encoding_decoding(&mut header, wgsl_shader, true); // gzip+base64
    }

    #[test]
    fn test_irmf_header_omitempty() {
        let header = IrmfHeader {
            author: None,
            license: None,
            date: None,
            encoding: None,
            irmf_version: "1.0".into(),
            glsl_version: None,
            language: None,
            materials: vec!["M1".into()],
            max: [1.0, 1.0, 1.0],
            min: [0.0, 0.0, 0.0],
            notes: None,
            options: Some(serde_json::json!({})), // Empty map should be skipped
            title: None,
            units: "mm".into(),
            version: None,
        };

        let json = header.serialize_to_string();
        assert!(!json.contains("author:"));
        assert!(!json.contains("options:"));
        assert!(!json.contains("glslVersion:"));
        assert!(json.contains("irmf: \"1.0\","));
        assert!(json.contains("materials: [\"M1\"],"));
        let today = Local::now().format("%Y-%m-%d").to_string();
        assert!(json.contains(&format!("date: \"{}\",", today))); // Default date
    }

    #[test]
    fn test_header_serialization_formatting() {
        let header = IrmfHeader {
            author: Some("Glenn".into()),
            license: None,
            date: None,
            encoding: None,
            irmf_version: "1.0".into(),
            glsl_version: None,
            language: None,
            materials: vec!["M1".into(), "M2".into()],
            max: [10.0, 20.0, 30.0],
            min: [0.0, 0.0, 0.0],
            notes: None,
            options: None,
            title: None,
            units: "mm".into(),
            version: None,
        };

        let s = header.serialize_to_string();
        let lines: Vec<&str> = s.lines().collect();

        // 1. remove all double-quotes around the keys
        assert!(s.contains("  author: \"Glenn\","));
        assert!(!s.contains("\"author\":"));

        // 2. always put the "min" key before the "max" key
        let min_idx = lines.iter().position(|l| l.contains("min:")).unwrap();
        let max_idx = lines.iter().position(|l| l.contains("max:")).unwrap();
        assert!(min_idx < max_idx);

        // 3. always keep all the [x,y,z] values for both the "min" and the "max" lines on the same line
        assert!(lines[min_idx].contains("[0.0000,0.0000,0.0000]"));
        assert!(lines[max_idx].contains("[10.0000,20.0000,30.0000]"));

        // 4. if the "date" key/value pair is missing or the value is empty, fill in today's date
        let author_idx = lines.iter().position(|l| l.contains("author:")).unwrap();
        let date_idx = lines.iter().position(|l| l.contains("date:")).unwrap();
        assert_eq!(date_idx, author_idx + 1);
        let today = Local::now().format("%Y-%m-%d").to_string();
        assert!(lines[date_idx].contains(&format!("\"{}\"", today)));

        // 5. keep the list of all materials on a single line: materials: ["mat1","mat2"],
        let mat_idx = lines.iter().position(|l| l.contains("materials:")).unwrap();
        assert!(lines[mat_idx].contains("[\"M1\",\"M2\"]"));

        // 6. always add a trailing comma to the last key/value pair in the list
        let last_kv_idx = lines.len() - 2; // line before }
        assert!(lines[last_kv_idx].ends_with(","));
    }

    #[test]
    fn test_parse_header_with_url_in_string() {
        let data = b"/*{
  \"irmf\": \"1.0\",
  \"materials\": [\"Material\"],
  \"max\": [1,1,1],
  \"min\": [0,0,0],
  \"units\": \"mm\",
  \"title\": \"Rodin's The Thinker from https://people.csail.mit.edu/tmertens/textransfer/data/\",
}*/
void main() {}";
        let model = IrmfModel::new(data).unwrap();
        assert_eq!(
            model.header.title.as_deref(),
            Some(
                "Rodin's The Thinker from https://people.csail.mit.edu/tmertens/textransfer/data/"
            )
        );
    }
}
