use serde::{Deserialize, Serialize};
use std::io::Read;
use flate2::read::GzDecoder;
use base64::{prelude::BASE64_STANDARD, Engine};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IrmfError {
    #[error("Unable to find leading '/*{{'")]
    MissingLeadingComment,
    #[error("Unable to find trailing '}}*/'")]
    MissingTrailingComment,
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Base64 decoding error: {0}")]
    Base64Error(#[from] base64::DecodeError),
    #[error("Gzip decoding error: {0}")]
    GzipError(#[from] std::io::Error),
    #[error("Unsupported IRMF version: {0}")]
    UnsupportedVersion(String),
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),
    #[error("Invalid materials: {0}")]
    InvalidMaterials(String),
    #[error("Invalid MBB: {0}")]
    InvalidMbb(String),
    #[error("Unsupported encoding: {0}")]
    UnsupportedEncoding(String),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IrmfHeader {
    pub author: Option<String>,
    pub license: Option<String>,
    pub date: Option<String>,
    pub encoding: Option<String>,
    #[serde(rename = "irmf")]
    pub irmf_version: String,
    pub glsl_version: Option<String>,
    pub language: Option<String>, // Default is "glsl"
    pub materials: Vec<String>,
    pub max: [f32; 3],
    pub min: [f32; 3],
    pub notes: Option<String>,
    pub options: Option<serde_json::Value>,
    pub title: Option<String>,
    pub units: String,
    pub version: Option<String>,
}

#[derive(Debug)]
pub struct IrmfModel {
    pub header: IrmfHeader,
    pub shader: String,
}

impl IrmfModel {
    pub fn new(data: &[u8]) -> Result<Self, IrmfError> {
        let (header, shader_payload) = parse_header(data)?;
        let shader = decode_shader(&header, shader_payload)?;
        
        let model = IrmfModel { header, shader };
        model.validate()?;
        
        Ok(model)
    }

    fn validate(&self) -> Result<(), IrmfError> {
        if self.header.irmf_version != "1.0" {
            return Err(IrmfError::UnsupportedVersion(self.header.irmf_version.clone()));
        }
        if self.header.materials.is_empty() {
            return Err(IrmfError::InvalidMaterials("Must list at least one material name".into()));
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

fn parse_header(data: &[u8]) -> Result<(IrmfHeader, &[u8]), IrmfError> {
    let start_tag = b"/*{";
    let end_tag = b"}*/";

    if !data.starts_with(start_tag) {
        return Err(IrmfError::MissingLeadingComment);
    }

    // Find the end tag. Note: Go used "\n}*/\n" but IRMF spec is usually "}*/"
    // Let's be a bit flexible but look for the first "}*/"
    let end_index = data.windows(end_tag.len())
        .position(|window| window == end_tag)
        .ok_or(IrmfError::MissingTrailingComment)?;

    let json_bytes = &data[2..end_index + 1]; // From "{" to "}"
    let header: IrmfHeader = serde_json::from_slice(json_bytes)?;

    let payload = &data[end_index + end_tag.len()..];
    // Trim leading whitespace/newlines from payload
    let mut start = 0;
    while start < payload.len() && (payload[start] as char).is_whitespace() {
        start += 1;
    }

    Ok((header, &payload[start..]))
}

fn decode_shader(header: &IrmfHeader, payload: &[u8]) -> Result<String, IrmfError> {
    match header.encoding.as_deref() {
        Some("gzip") => {
            let mut decoder = GzDecoder::new(payload);
            let mut shader = String::new();
            decoder.read_to_string(&mut shader).map_err(IrmfError::GzipError)?;
            Ok(shader)
        }
        Some("gzip+base64") => {
            // Trim whitespace from base64 payload
            let payload_str = std::str::from_utf8(payload).unwrap_or("").trim();
            let decoded = BASE64_STANDARD.decode(payload_str)?;
            let mut decoder = GzDecoder::new(&decoded[..]);
            let mut shader = String::new();
            decoder.read_to_string(&mut shader).map_err(IrmfError::GzipError)?;
            Ok(shader)
        }
        None | Some("") => {
            Ok(String::from_utf8_lossy(payload).into_owned())
        }
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
}
