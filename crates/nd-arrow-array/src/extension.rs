//! The registered Arrow extension type of an nd-encoded column.
//!
//! The storage type is a `Struct`. Each row holds one nd array:
//!
//! ```text
//! Struct{
//!   values:    List<T>,       // the flat, C-order values
//!   dim_sizes: List<UInt32>,  // size per axis
//!   dim_names: List<Utf8>,    // name per axis
//! }
//! ```
//!
//! The extension metadata is JSON. It holds the encoding version and the
//! static [`AxisMeta`] of the axes, so a plan can read the metadata at plan
//! time.

use arrow_schema::extension::ExtensionType;
use arrow_schema::{ArrowError, DataType, Fields};
use serde::{Deserialize, Serialize};

use crate::axis::{AxisMeta, AxisOrder};
use crate::dimensions::Dimensions;
use crate::error::Result;

/// The current version of the nd encoding.
pub const ND_ENCODING_VERSION: u32 = 1;

/// The JSON metadata of an [`NdArrayType`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NdArrayMetadata {
    #[serde(default = "default_version")]
    pub version: u32,
    /// Metadata of the axes that have it. Other axes are absent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub axes: Vec<AxisEntry>,
}

/// The serialized [`AxisMeta`] of one named axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisEntry {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coordinate: Option<String>,
    #[serde(default)]
    pub order: AxisOrder,
}

impl NdArrayMetadata {
    /// Metadata that records the [`AxisMeta`] of each axis in `dims`.
    pub fn from_dims(dims: &Dimensions) -> Self {
        let axes = dims
            .iter()
            .filter_map(|dim| {
                dim.meta().map(|meta| AxisEntry {
                    name: dim.name().to_string(),
                    coordinate: meta.coordinate_column().map(str::to_string),
                    order: meta.order(),
                })
            })
            .collect();
        Self {
            version: ND_ENCODING_VERSION,
            axes,
        }
    }

    /// The [`AxisMeta`] recorded for the axis `name`.
    pub fn axis_meta(&self, name: &str) -> Option<AxisMeta> {
        self.axes
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| AxisMeta::new(entry.coordinate.as_deref().map(Into::into), entry.order))
    }
}

fn default_version() -> u32 {
    ND_ENCODING_VERSION
}

impl Default for NdArrayMetadata {
    fn default() -> Self {
        Self {
            version: ND_ENCODING_VERSION,
            axes: Vec::new(),
        }
    }
}

/// The Arrow extension type of an nd-encoded column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdArrayType {
    value_type: DataType,
    metadata: NdArrayMetadata,
}

impl NdArrayType {
    /// An nd extension type for columns with elements of `value_type`.
    pub fn new(value_type: DataType) -> Self {
        Self::with_metadata(value_type, NdArrayMetadata::default())
    }

    pub fn with_metadata(value_type: DataType, metadata: NdArrayMetadata) -> Self {
        Self {
            value_type,
            metadata,
        }
    }

    /// The element type of the `values` list.
    pub fn value_type(&self) -> &DataType {
        &self.value_type
    }
}

impl ExtensionType for NdArrayType {
    const NAME: &'static str = "nd.array";

    type Metadata = NdArrayMetadata;

    fn metadata(&self) -> &Self::Metadata {
        &self.metadata
    }

    fn serialize_metadata(&self) -> Option<String> {
        Some(serde_json::to_string(&self.metadata).expect("nd metadata serializes"))
    }

    fn deserialize_metadata(metadata: Option<&str>) -> Result<Self::Metadata> {
        let metadata = match metadata.map(str::trim) {
            None | Some("") => NdArrayMetadata::default(),
            Some(json) => serde_json::from_str(json).map_err(|e| {
                ArrowError::InvalidArgumentError(format!("invalid nd extension metadata: {e}"))
            })?,
        };
        if metadata.version != ND_ENCODING_VERSION {
            return Err(ArrowError::InvalidArgumentError(format!(
                "unsupported nd encoding version {} (expected {ND_ENCODING_VERSION})",
                metadata.version
            )));
        }
        Ok(metadata)
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<()> {
        let value_type = storage_value_type(data_type)?;
        if value_type != self.value_type {
            return Err(ArrowError::InvalidArgumentError(format!(
                "nd storage carries {value_type} values but the extension type expects {}",
                self.value_type
            )));
        }
        Ok(())
    }

    fn try_new(data_type: &DataType, metadata: Self::Metadata) -> Result<Self> {
        Ok(Self::with_metadata(
            storage_value_type(data_type)?,
            metadata,
        ))
    }
}

/// Check the storage layout and return the element type of `values`.
fn storage_value_type(data_type: &DataType) -> Result<DataType> {
    let DataType::Struct(fields) = data_type else {
        return Err(invalid(format!(
            "nd storage must be a Struct, got {data_type}"
        )));
    };
    let values = list_item(fields, "values")?;
    let sizes = list_item(fields, "dim_sizes")?;
    let names = list_item(fields, "dim_names")?;
    if sizes != DataType::UInt32 {
        return Err(invalid(format!(
            "nd 'dim_sizes' must be List<UInt32>, got List<{sizes}>"
        )));
    }
    if names != DataType::Utf8 {
        return Err(invalid(format!(
            "nd 'dim_names' must be List<Utf8>, got List<{names}>"
        )));
    }
    Ok(values)
}

fn list_item(fields: &Fields, name: &str) -> Result<DataType> {
    let field = fields
        .iter()
        .find(|f| f.name() == name)
        .ok_or_else(|| invalid(format!("nd storage is missing a '{name}' field")))?;
    match field.data_type() {
        DataType::List(item) => Ok(item.data_type().clone()),
        other => Err(invalid(format!("nd '{name}' must be a List, got {other}"))),
    }
}

fn invalid(message: String) -> ArrowError {
    ArrowError::InvalidArgumentError(message)
}

#[cfg(test)]
mod tests {
    use arrow_schema::Field;
    use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};

    use super::*;
    use crate::encoding::nd_encoded_type;

    fn encoded_field() -> Field {
        Field::new("sst", nd_encoded_type(&DataType::Float64), true)
            .with_extension_type(NdArrayType::new(DataType::Float64))
    }

    #[test]
    fn field_round_trip() {
        let field = encoded_field();
        assert_eq!(
            field
                .metadata()
                .get(EXTENSION_TYPE_NAME_KEY)
                .map(String::as_str),
            Some("nd.array")
        );
        let ext = field.try_extension_type::<NdArrayType>().unwrap();
        assert_eq!(ext.value_type(), &DataType::Float64);
        assert_eq!(ext.metadata().version, ND_ENCODING_VERSION);
    }

    #[test]
    fn missing_metadata_uses_the_default() {
        let field = Field::new("sst", nd_encoded_type(&DataType::Int32), true)
            .with_metadata([(EXTENSION_TYPE_NAME_KEY.to_string(), "nd.array".to_string())].into());
        let ext = field.try_extension_type::<NdArrayType>().unwrap();
        assert_eq!(ext.metadata(), &NdArrayMetadata::default());
    }

    #[test]
    fn unknown_version_is_rejected() {
        let mut field = encoded_field();
        let mut metadata = field.metadata().clone();
        metadata.insert(
            EXTENSION_TYPE_METADATA_KEY.to_string(),
            r#"{"version":99}"#.to_string(),
        );
        field = field.with_metadata(metadata);
        assert!(field.try_extension_type::<NdArrayType>().is_err());
    }

    #[test]
    fn axis_meta_round_trips_through_the_field() {
        use crate::dimensions::Dimension;

        let dims = Dimensions::try_new(vec![
            Dimension::new("time", 2)
                .with_meta(Some(AxisMeta::coordinate("time", AxisOrder::Ascending))),
            Dimension::new("N_PROF", 3).with_meta(Some(AxisMeta::no_coordinate())),
            Dimension::new("lon", 4),
        ])
        .unwrap();
        let ext = NdArrayType::with_metadata(DataType::Float64, NdArrayMetadata::from_dims(&dims));
        let field =
            Field::new("sst", nd_encoded_type(&DataType::Float64), true).with_extension_type(ext);

        let metadata = field
            .try_extension_type::<NdArrayType>()
            .unwrap()
            .metadata()
            .clone();
        assert_eq!(metadata.axes.len(), 2);
        assert_eq!(
            metadata.axis_meta("time"),
            Some(AxisMeta::coordinate("time", AxisOrder::Ascending))
        );
        assert_eq!(
            metadata.axis_meta("N_PROF"),
            Some(AxisMeta::no_coordinate())
        );
        assert_eq!(metadata.axis_meta("lon"), None);
    }

    #[test]
    fn metadata_without_axes_stays_small() {
        let json = NdArrayType::new(DataType::Int32)
            .serialize_metadata()
            .unwrap();
        assert_eq!(json, r#"{"version":1}"#);
    }

    #[test]
    fn a_plain_type_is_not_supported() {
        let ext = NdArrayType::new(DataType::Float64);
        assert!(ext.supports_data_type(&DataType::Float64).is_err());
        assert!(NdArrayType::try_new(&DataType::Float64, NdArrayMetadata::default()).is_err());
    }

    #[test]
    fn a_value_type_mismatch_is_not_supported() {
        let ext = NdArrayType::new(DataType::Float64);
        assert!(
            ext.supports_data_type(&nd_encoded_type(&DataType::Int32))
                .is_err()
        );
        assert!(
            ext.supports_data_type(&nd_encoded_type(&DataType::Float64))
                .is_ok()
        );
    }

    #[test]
    fn wrong_dimension_list_types_are_rejected() {
        let DataType::Struct(fields) = nd_encoded_type(&DataType::Float64) else {
            unreachable!()
        };
        let mut fields: Vec<Field> = fields.iter().map(|f| f.as_ref().clone()).collect();
        fields[1] = Field::new(
            "dim_sizes",
            DataType::List(std::sync::Arc::new(Field::new(
                "item",
                DataType::Int64,
                false,
            ))),
            false,
        );
        let bogus = DataType::Struct(fields.into());
        assert!(NdArrayType::try_new(&bogus, NdArrayMetadata::default()).is_err());
    }
}
