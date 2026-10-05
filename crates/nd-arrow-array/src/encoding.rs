//! A self-describing Arrow encoding of nd arrays.
//!
//! An [`NdArrowArray`] is encoded as an Arrow `Struct` whose single row carries
//! the flat values plus the dimensions:
//!
//! ```text
//! Struct{
//!   values:    List<T>,       // the flat, C-order values (one list element)
//!   dim_sizes: List<UInt32>,  // size per axis
//!   dim_names: List<Utf8>,    // name per axis
//! }
//! ```
//!
//! The struct field carries the registered [`NdArrayType`] extension type, so
//! the intent survives IPC. Because it is an ordinary Arrow array, an nd batch encoded
//! this way rides through a `DataSourceExec` (and any other operator) as a
//! normal `RecordBatch`; `NdSourceExec` decodes
//! it back into an [`NdRecordBatch`] on the way out.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::nd_err;
use crate::error::{ArrowError, Result};
use arrow::array::{
    Array, ArrayRef, ListArray, StringArray, StructArray, UInt32Array, new_null_array,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};

use arrow_schema::extension::ExtensionType;

use super::array::NdArrowArray;
use super::batch::NdRecordBatch;
use super::dimensions::{Dimension, Dimensions};
use super::extension::{NdArrayMetadata, NdArrayType, storage_value_type};

fn err(e: impl std::fmt::Display) -> ArrowError {
    ArrowError::InvalidArgumentError(e.to_string())
}

/// The three struct fields of the nd encoding for a given element type.
fn nd_struct_fields(value_type: &DataType) -> Fields {
    Fields::from(vec![
        Field::new(
            "values",
            DataType::List(Arc::new(Field::new("item", value_type.clone(), true))),
            false,
        ),
        Field::new(
            "dim_sizes",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
            false,
        ),
        Field::new(
            "dim_names",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),
            false,
        ),
    ])
}

/// The struct `DataType` of the nd encoding for a given element type.
pub fn nd_encoded_type(value_type: &DataType) -> DataType {
    DataType::Struct(nd_struct_fields(value_type))
}

/// `field`, with its values carried as an [`NdArrayType`] struct.
///
/// The field's own metadata travels with it, and the extension keys go on top.
/// A scan's target schema is the *encoded* one, and the GeoArrow keys of a
/// geometry column live in that metadata.
pub fn nd_encoded_field_of(field: &Field) -> Field {
    encoded_field(field, NdArrayType::new(field.data_type().clone()))
}

/// Like [`nd_encoded_field_of`], but the extension metadata also records the
/// axis names of `dims` and the [`AxisMeta`](crate::AxisMeta) of each axis
/// that has it. A format that knows the axes of each column at plan time uses
/// this, so the plan can derive the grid and its sort order.
pub fn nd_encoded_field_with_dims(field: &Field, dims: &Dimensions) -> Field {
    let metadata = NdArrayMetadata::from_dims(dims);
    encoded_field(
        field,
        NdArrayType::with_metadata(field.data_type().clone(), metadata),
    )
}

fn encoded_field(field: &Field, extension: NdArrayType) -> Field {
    Field::new(field.name(), nd_encoded_type(field.data_type()), true)
        .with_metadata(field.metadata().clone())
        .with_extension_type(extension)
}

/// The extension metadata of an nd-encoded field, or `None` when the field is
/// not nd-encoded.
pub fn nd_field_metadata(field: &Field) -> Option<NdArrayMetadata> {
    field
        .try_extension_type::<NdArrayType>()
        .ok()
        .map(|ext| ext.metadata().clone())
}

/// The nd-encoded schema of a logical schema: every field becomes an nd
/// struct column of the same name. This is the schema a `DataSourceExec` carries
/// while nd data flows up to `NdSourceExec`.
pub fn encoded_schema(logical: &Schema) -> Schema {
    let fields: Vec<Field> = logical
        .fields()
        .iter()
        .map(|f| nd_encoded_field_of(f))
        .collect();
    Schema::new_with_metadata(fields, logical.metadata().clone())
}

/// True when `field` is an nd-encoded column: it carries the [`NdArrayType`]
/// extension type over a valid storage type.
pub fn is_nd_encoded(field: &Field) -> bool {
    field.try_extension_type::<NdArrayType>().is_ok()
}

/// Element type carried by an nd-encoded struct type (the `values` list item).
pub fn nd_value_type(encoded: &DataType) -> Result<DataType> {
    storage_value_type(encoded)
}

/// Encode one [`NdArrowArray`] as a single-row `Struct` array.
pub fn encode_nd_array(array: &NdArrowArray) -> ArrayRef {
    let values = array.values();
    let dims = array.dims();

    let values_list = ListArray::new(
        Arc::new(Field::new("item", values.data_type().clone(), true)),
        OffsetBuffer::from_lengths([values.len()]),
        values.clone(),
        None,
    );

    let sizes: UInt32Array = dims.iter().map(|d| d.size() as u32).collect();
    let dim_sizes_list = ListArray::new(
        Arc::new(Field::new("item", DataType::UInt32, false)),
        OffsetBuffer::from_lengths([dims.rank()]),
        Arc::new(sizes),
        None,
    );

    let names: StringArray = dims.iter().map(|d| Some(d.name().to_string())).collect();
    let dim_names_list = ListArray::new(
        Arc::new(Field::new("item", DataType::Utf8, false)),
        OffsetBuffer::from_lengths([dims.rank()]),
        Arc::new(names),
        None,
    );

    let struct_array = StructArray::new(
        nd_struct_fields(values.data_type()),
        vec![
            Arc::new(values_list),
            Arc::new(dim_sizes_list),
            Arc::new(dim_names_list),
        ],
        None,
    );
    Arc::new(struct_array)
}

/// Decode row `row` of an nd-encoded `Struct` column back into an [`NdArrowArray`].
///
/// A null struct row (a column a file lacked, null-filled by the schema adapter)
/// decodes to a rank-0 null scalar, which broadcasts to an all-null column.
pub fn decode_nd_array(column: &ArrayRef, row: usize) -> Result<NdArrowArray> {
    let structs = column
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| err("nd column is not a Struct array"))?;

    if structs.is_null(row) {
        let value_type = nd_value_type(structs.data_type())?;
        return NdArrowArray::try_new(new_null_array(&value_type, 1), Dimensions::scalar());
    }

    let list_at = |name: &str| -> Result<ArrayRef> {
        let list = structs
            .column_by_name(name)
            .ok_or_else(|| err(format!("nd struct is missing field '{name}'")))?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| err(format!("nd struct field '{name}' is not a List")))?;
        Ok(list.value(row))
    };

    let values = list_at("values")?;

    let sizes = list_at("dim_sizes")?;
    let sizes = sizes
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| err("nd 'dim_sizes' is not UInt32"))?;

    let names = list_at("dim_names")?;
    let names = names
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| err("nd 'dim_names' is not Utf8"))?;

    if sizes.len() != names.len() {
        return nd_err!(
            "nd dim_sizes ({}) and dim_names ({}) disagree",
            sizes.len(),
            names.len()
        );
    }

    let dims = Dimensions::try_new(
        (0..sizes.len())
            .map(|i| Dimension::new(names.value(i), sizes.value(i) as usize))
            .collect(),
    )?;

    NdArrowArray::try_new(values, dims)
}

/// Encode an [`NdRecordBatch`] as a flat `RecordBatch` of nd-encoded (`nd.array`)
/// struct columns — one struct row per column.
pub fn encode_nd_record_batch(batch: &NdRecordBatch) -> Result<RecordBatch> {
    let fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, column)| {
            nd_encoded_field_with_dims(
                &Field::new(field.name(), column.data_type().clone(), true),
                column.dims(),
            )
        })
        .collect();

    let columns: Vec<ArrayRef> = batch.columns().iter().map(encode_nd_array).collect();

    let options = RecordBatchOptions::new().with_row_count(Some(1));
    RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), columns, &options)
}

/// The logical (decoded) schema of an nd-encoded schema: each `nd.array`
/// struct column becomes its element type.
///
/// Nullability follows the encoded field. [`nd_encoded_field_of`] makes every
/// column of a file nullable, because a file a collection is not obliged to be
/// uniform over is null-filled for the columns it lacks. A `PARTITIONED BY`
/// column is not read from a file at all — its value is in the path, and the
/// scan always has it — so a format may encode it as the table declared it, and
/// the table gets that column back exactly as it declared it.
///
/// When the encoded field records its axes, the logical field carries the nd
/// metadata under [`ND_METADATA_KEY`], so a logical plan can see the axes of
/// each column. See [`nd_logical_metadata`].
pub fn logical_schema(encoded: &Schema) -> Result<SchemaRef> {
    let fields = encoded
        .fields()
        .iter()
        .map(|field| {
            let logical = Field::new(
                field.name(),
                nd_value_type(field.data_type())?,
                field.is_nullable(),
            );
            let metadata = nd_field_metadata(field)
                .filter(|m| m.dims.is_some() || !m.axes.is_empty())
                .and_then(|m| serde_json::to_string(&m).ok())
                .or_else(|| field.metadata().get(ND_METADATA_KEY).cloned());
            Ok(match metadata {
                Some(json) => {
                    logical.with_metadata(HashMap::from([(ND_METADATA_KEY.to_string(), json)]))
                }
                None => logical,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(Schema::new(fields)))
}

/// The field metadata key of the nd metadata of a logical field.
pub const ND_METADATA_KEY: &str = "nd.array.metadata";

/// The nd metadata of a logical field: its axis names and axis entries, or
/// `None` when the format did not record them.
pub fn nd_logical_metadata(field: &Field) -> Option<NdArrayMetadata> {
    serde_json::from_str(field.metadata().get(ND_METADATA_KEY)?).ok()
}

/// The coordinate column of each axis of a logical schema, as `(axis, column)`
/// in the order of first use.
///
/// An axis entry names the column, or has no column and turns the rule off.
/// Without an entry, the field with the name of the axis is the coordinate.
/// The column counts only when it lies on that axis alone.
pub fn schema_coordinates(schema: &Schema) -> Vec<(String, String)> {
    let metadata: Vec<(&str, NdArrayMetadata)> = schema
        .fields()
        .iter()
        .filter_map(|f| Some((f.name().as_str(), nd_logical_metadata(f)?)))
        .collect();
    let mut axes: Vec<&String> = Vec::new();
    for name in metadata.iter().flat_map(|(_, m)| m.dims.iter().flatten()) {
        if !axes.contains(&name) {
            axes.push(name);
        }
    }
    let dims_of = |column: &str| {
        metadata
            .iter()
            .find(|(name, _)| *name == column)
            .and_then(|(_, m)| m.dims.clone())
    };
    axes.into_iter()
        .filter_map(|axis| {
            let entry = metadata
                .iter()
                .flat_map(|(_, m)| &m.axes)
                .find(|e| &e.name == axis);
            let column = match entry {
                Some(entry) => entry.coordinate.clone()?,
                None => axis.clone(),
            };
            (dims_of(&column) == Some(vec![axis.clone()])).then(|| (axis.clone(), column))
        })
        .collect()
}

/// How many nd batches an encoded `RecordBatch` carries.
///
/// One per row, except for a zero-column `COUNT(*)`-style carrier, whose rows
/// are the payload rather than an nd-array-per-row.
pub fn nd_batch_count(batch: &RecordBatch) -> usize {
    if batch.num_columns() == 0 {
        1
    } else {
        batch.num_rows()
    }
}

/// Decode a single row of an nd-encoded `RecordBatch` into an [`NdRecordBatch`].
/// The target grid is inferred as the union of the columns' dimensions, ordered
/// by the highest-rank column.
pub fn decode_nd_record_batch_row(batch: &RecordBatch, row: usize) -> Result<NdRecordBatch> {
    // A zero-column batch is a COUNT(*)-style row carrier: preserve its row
    // count via a synthetic one-axis grid so the broadcast reproduces it.
    if batch.num_columns() == 0 {
        let target = Dimensions::try_new(vec![Dimension::new("row", batch.num_rows())])?;
        return NdRecordBatch::try_new(Arc::new(Schema::empty()), vec![], target);
    }

    let columns = batch
        .columns()
        .iter()
        .zip(batch.schema_ref().fields())
        .map(|(column, field)| {
            let array = decode_nd_array(column, row)?;
            match nd_field_metadata(field) {
                Some(metadata) if !metadata.axes.is_empty() => {
                    let dims = array.dims().with_axis_meta(|name| metadata.axis_meta(name));
                    NdArrowArray::try_new(array.values().clone(), dims)
                }
                _ => Ok(array),
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let target = infer_target(&columns)?;
    let schema = logical_schema(batch.schema_ref())?;
    NdRecordBatch::try_new(schema, columns, target)
}

/// The axis names of the grid of a scan with the encoded schema `encoded`,
/// outer first, from the field metadata. The rule is the rule of
/// [`infer_target`]. `None` when a field does not record its axes.
pub fn plan_target_axes(encoded: &Schema) -> Option<Vec<String>> {
    let columns = encoded
        .fields()
        .iter()
        .map(|field| nd_field_metadata(field)?.dims)
        .collect::<Option<Vec<_>>>()?;
    Some(grid_order(&columns, |name| name))
}

/// Infer the target grid from decoded columns: the highest-rank column defines
/// the axis order (it spans the grid in C-order), and any axis only present on
/// lower-rank columns is appended.
///
/// A file opener that builds nd batches itself uses the same rule, so a batch
/// it emits and a batch the decoder rebuilds agree on the grid.
pub fn infer_target(columns: &[NdArrowArray]) -> Result<Dimensions> {
    let axes: Vec<Vec<Dimension>> = columns
        .iter()
        .map(|c| c.dims().iter().cloned().collect())
        .collect();
    Dimensions::try_new(grid_order(&axes, |dim| dim.name()))
}

/// The axes of the widest column, then each other axis in order of first use.
fn grid_order<T: Clone>(columns: &[Vec<T>], name: impl Fn(&T) -> &str) -> Vec<T> {
    let mut order: Vec<T> = columns
        .iter()
        .max_by_key(|axes| axes.len())
        .cloned()
        .unwrap_or_default();
    for axis in columns.iter().flatten() {
        if !order.iter().any(|known| name(known) == name(axis)) {
            order.push(axis.clone());
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{AsArray, Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Float64Type, Int32Type, Schema};

    use super::*;

    fn dims(spec: &[(&str, usize)]) -> Dimensions {
        Dimensions::try_new(
            spec.iter()
                .map(|(name, size)| Dimension::new(*name, *size))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn nd_array_round_trip() {
        let a = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            dims(&[("time", 2), ("lat", 3)]),
        )
        .unwrap();

        let encoded = encode_nd_array(&a);
        // The encoded column is a single struct row.
        assert_eq!(encoded.len(), 1);

        let decoded = decode_nd_array(&encoded, 0).unwrap();
        assert_eq!(decoded.dims(), a.dims());
        assert_eq!(
            decoded.values().as_primitive::<Int32Type>().values(),
            &[1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn encoded_field_is_tagged() {
        let field = nd_encoded_field_of(&Field::new("sst", DataType::Float64, true));
        assert!(is_nd_encoded(&field));
        assert_eq!(nd_value_type(field.data_type()).unwrap(), DataType::Float64);
    }

    #[test]
    fn a_field_without_the_extension_type_is_not_encoded() {
        let plain = Field::new("sst", nd_encoded_type(&DataType::Float64), true);
        assert!(!is_nd_encoded(&plain));

        let other_name = plain
            .with_metadata([("ARROW:extension:name".to_string(), "beacon.nd".to_string())].into());
        assert!(!is_nd_encoded(&other_name));
    }

    #[test]
    fn encoded_field_keeps_its_own_metadata() {
        let field = Field::new("geom", DataType::Binary, true)
            .with_metadata([("crs".to_string(), "EPSG:4326".to_string())].into());
        let encoded = nd_encoded_field_of(&field);
        assert!(is_nd_encoded(&encoded));
        assert_eq!(
            encoded.metadata().get("crs").map(String::as_str),
            Some("EPSG:4326")
        );
    }

    #[test]
    fn axis_meta_survives_the_record_batch_encoding() {
        use crate::axis::{AxisMeta, AxisOrder};

        let time_dim = Dimension::new("time", 2)
            .with_meta(Some(AxisMeta::coordinate("time", AxisOrder::Descending)));
        let lat_dim = Dimension::new("lat", 3);
        let grid = Dimensions::try_new(vec![time_dim.clone(), lat_dim.clone()]).unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("time", DataType::Int32, true),
            Field::new("sst", DataType::Float64, true),
        ]));
        let time = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![8, 7])),
            Dimensions::try_new(vec![time_dim]).unwrap(),
        )
        .unwrap();
        let sst = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2])),
            grid.clone(),
        )
        .unwrap();
        let nd = NdRecordBatch::try_new(schema, vec![time, sst], grid.clone()).unwrap();

        let decoded = decode_nd_record_batch_row(&encode_nd_record_batch(&nd).unwrap(), 0).unwrap();
        assert_eq!(decoded.target(), &grid);
        assert_eq!(decoded.target().get(0).order(), AxisOrder::Descending);
        assert_eq!(
            decoded.column(0).dims().get(0).order(),
            AxisOrder::Descending
        );
        assert_eq!(decoded.target().get(1).meta(), None);
    }

    #[test]
    fn a_projected_batch_has_the_grid_of_the_selected_columns() {
        // Select only `elev{lat}`: the grid is `lat`, and `time` is gone.
        let schema = Arc::new(Schema::new(vec![
            Field::new("sst", DataType::Float64, true),
            Field::new("elev", DataType::Int32, true),
        ]));
        let sst = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2])),
            dims(&[("time", 2), ("lat", 3)]),
        )
        .unwrap();
        let elev = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![5, 10, 15])),
            dims(&[("lat", 3)]),
        )
        .unwrap();
        let nd = NdRecordBatch::try_new(schema, vec![sst, elev], dims(&[("time", 2), ("lat", 3)]))
            .unwrap();
        let encoded = encode_nd_record_batch(&nd).unwrap();

        let decoded = decode_nd_record_batch_row(&encoded.project(&[1]).unwrap(), 0).unwrap();
        assert_eq!(decoded.target(), &dims(&[("lat", 3)]));
        assert_eq!(decoded.num_rows(), 3);
    }

    #[test]
    fn the_plan_grid_follows_the_widest_column() {
        let field = |name: &str, axes: &[(&str, usize)]| {
            nd_encoded_field_with_dims(&Field::new(name, DataType::Float64, true), &dims(axes))
        };
        let schema = Schema::new(vec![
            field("lat", &[("lat", 3)]),
            field("sst", &[("time", 2), ("lat", 3)]),
            field("depth", &[("depth", 4)]),
        ]);
        assert_eq!(
            plan_target_axes(&schema),
            Some(vec![
                "time".to_string(),
                "lat".to_string(),
                "depth".to_string()
            ])
        );

        // A field without its axes gives no plan grid.
        let schema = Schema::new(vec![nd_encoded_field_of(&Field::new(
            "lat",
            DataType::Float64,
            true,
        ))]);
        assert_eq!(plan_target_axes(&schema), None);
    }

    #[test]
    fn record_batch_round_trip_infers_target() {
        // time coord (1-D), lat coord (1-D), sst data (2-D) over (time=2, lat=3).
        let schema = Arc::new(Schema::new(vec![
            Field::new("time", DataType::Int32, true),
            Field::new("lat", DataType::Int32, true),
            Field::new("sst", DataType::Float64, true),
        ]));
        let time =
            NdArrowArray::try_new(Arc::new(Int32Array::from(vec![7, 8])), dims(&[("time", 2)]))
                .unwrap();
        let lat = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![10, 20, 30])),
            dims(&[("lat", 3)]),
        )
        .unwrap();
        let sst = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2])),
            dims(&[("time", 2), ("lat", 3)]),
        )
        .unwrap();
        let nd = NdRecordBatch::try_new(
            schema.clone(),
            vec![time, lat, sst],
            dims(&[("time", 2), ("lat", 3)]),
        )
        .unwrap();

        // Encode → flat struct RecordBatch, all columns tagged nd.array.
        let encoded = encode_nd_record_batch(&nd).unwrap();
        assert_eq!(encoded.num_rows(), 1);
        assert_eq!(encoded.num_columns(), 3);
        for field in encoded.schema().fields() {
            assert!(is_nd_encoded(field), "{} not tagged", field.name());
        }

        // Decode → NdRecordBatch, target inferred from the widest (sst) column.
        let decoded = decode_nd_record_batch_row(&encoded, 0).unwrap();
        assert_eq!(decoded.target(), &dims(&[("time", 2), ("lat", 3)]));

        // Materializing the decoded batch matches the original.
        let expected = nd.materialize().unwrap();
        let actual = decoded.materialize().unwrap();
        // The decoded schema also carries the axes of each column.
        assert_eq!(actual.columns(), expected.columns());
        assert_eq!(
            actual.column(2).as_primitive::<Float64Type>().values(),
            &[0.0, 0.1, 0.2, 1.0, 1.1, 1.2]
        );
    }

    #[test]
    fn null_struct_column_decodes_to_null_scalar() {
        // A missing (null-filled) nd column decodes to a rank-0 null scalar and
        // broadcasts to an all-null column.
        let null_struct = arrow::array::new_null_array(&nd_encoded_type(&DataType::Float64), 1);
        let decoded = decode_nd_array(&null_struct, 0).unwrap();
        assert_eq!(decoded.dims().rank(), 0);
        let target = dims(&[("time", 3)]);
        let out = decoded.materialize(&target).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out.null_count(), 3);
    }

    #[test]
    fn zero_column_batch_preserves_row_count() {
        // COUNT(*)-style zero-column batch → nd batch that materializes to the
        // same row count with no columns.
        let empty = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(42)),
        )
        .unwrap();
        let nd = decode_nd_record_batch_row(&empty, 0).unwrap();
        let flat = nd.materialize().unwrap();
        assert_eq!(flat.num_columns(), 0);
        assert_eq!(flat.num_rows(), 42);
    }

    #[test]
    fn nd_value_type_rejects_non_nd_types() {
        // A plain column type is not an nd encoding...
        assert!(nd_value_type(&DataType::Float64).is_err());
        // ...and neither is a struct whose `values` field is not a List.
        let bogus = DataType::Struct(Fields::from(vec![Field::new(
            "values",
            DataType::Float64,
            true,
        )]));
        assert!(nd_value_type(&bogus).is_err());
        // ...nor one missing `values` entirely.
        let missing = DataType::Struct(Fields::from(vec![Field::new(
            "other",
            DataType::Utf8,
            true,
        )]));
        assert!(nd_value_type(&missing).is_err());
    }

    #[test]
    fn target_appends_axes_only_present_on_narrow_columns() {
        // The widest column (sst{time,lat}) sets the axis order; `depth`, which
        // only a 1-D column carries, is appended rather than dropped — otherwise
        // that column could not broadcast onto the grid at all.
        let schema = Arc::new(Schema::new(vec![
            Field::new("sst", DataType::Float64, true),
            Field::new("depth", DataType::Int32, true),
        ]));
        let sst = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2])),
            dims(&[("time", 2), ("lat", 3)]),
        )
        .unwrap();
        let depth = NdArrowArray::try_new(
            Arc::new(Int32Array::from(vec![5, 10])),
            dims(&[("depth", 2)]),
        )
        .unwrap();
        let nd = NdRecordBatch::try_new(
            schema,
            vec![sst, depth],
            dims(&[("time", 2), ("lat", 3), ("depth", 2)]),
        )
        .unwrap();

        let decoded = decode_nd_record_batch_row(&encode_nd_record_batch(&nd).unwrap(), 0).unwrap();
        assert_eq!(
            decoded.target(),
            &dims(&[("time", 2), ("lat", 3), ("depth", 2)])
        );
        assert_eq!(decoded.materialize().unwrap().num_rows(), 12);
    }

    #[test]
    fn scalar_column_round_trips_as_rank_zero() {
        // A rank-0 attribute column encodes with empty dim lists and decodes
        // back to a scalar (not to a 1-element axis).
        let scalar = NdArrowArray::try_new(
            Arc::new(Float64Array::from(vec![42.0])),
            Dimensions::scalar(),
        )
        .unwrap();
        let decoded = decode_nd_array(&encode_nd_array(&scalar), 0).unwrap();
        assert_eq!(decoded.dims().rank(), 0);
        assert_eq!(decoded.values().len(), 1);
    }

    #[test]
    fn every_row_of_a_coalesced_encoded_batch_decodes() {
        // `encode_nd_record_batch` emits one row per nd batch, but any operator
        // that concatenates batches (RoundRobinBatch repartitioning coalesces
        // small ones) yields a multi-row encoded batch. Decoding row 0 alone
        // silently drops the rest, which loses whole files from a scan.
        let batch_of = |values: Vec<i32>, rows: usize, cols: usize| {
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
            let column = NdArrowArray::try_new(
                Arc::new(Int32Array::from(values)),
                dims(&[("time", rows), ("lat", cols)]),
            )
            .unwrap();
            let target = dims(&[("time", rows), ("lat", cols)]);
            NdRecordBatch::try_new(schema, vec![column], target).unwrap()
        };

        let first = encode_nd_record_batch(&batch_of(vec![1, 2, 3, 4, 5, 6], 2, 3)).unwrap();
        let second = encode_nd_record_batch(&batch_of(vec![7, 8, 9, 10], 2, 2)).unwrap();
        assert_eq!(first.num_rows(), 1);
        assert_eq!(second.num_rows(), 1);

        let merged = arrow::compute::concat_batches(&first.schema(), &[first, second]).unwrap();
        assert_eq!(nd_batch_count(&merged), 2);

        let rows: Vec<usize> = (0..nd_batch_count(&merged))
            .map(|r| decode_nd_record_batch_row(&merged, r).unwrap().num_rows())
            .collect();
        assert_eq!(rows, vec![6, 4], "both nd batches must survive the concat");
    }

    #[test]
    fn logical_schema_unwraps_structs() {
        let encoded = Schema::new(vec![
            nd_encoded_field_of(&Field::new("lat", DataType::Int32, true)),
            nd_encoded_field_of(&Field::new("sst", DataType::Float64, true)),
        ]);
        let logical = logical_schema(&encoded).unwrap();
        assert_eq!(logical.field(0).data_type(), &DataType::Int32);
        assert_eq!(logical.field(1).data_type(), &DataType::Float64);
    }

    /// A logical schema with one field per `(name, axes)`, all `Float64`.
    fn planned(
        columns: &[(&str, &[&str])],
        meta: impl Fn(&str) -> Option<crate::AxisMeta>,
    ) -> SchemaRef {
        let encoded: Vec<Field> = columns
            .iter()
            .map(|(name, axes)| {
                let dims = Dimensions::try_new(
                    axes.iter()
                        .map(|a| Dimension::new(*a, 1).with_meta(meta(a)))
                        .collect(),
                )
                .unwrap();
                nd_encoded_field_with_dims(&Field::new(*name, DataType::Float64, true), &dims)
            })
            .collect();
        logical_schema(&Schema::new(encoded)).unwrap()
    }

    #[test]
    fn the_logical_schema_carries_the_axes_of_each_column() {
        let schema = planned(&[("sst", &["time", "lat"])], |_| None);
        let metadata = nd_logical_metadata(schema.field(0)).unwrap();
        assert_eq!(
            metadata.dims,
            Some(vec!["time".to_string(), "lat".to_string()])
        );
        // The metadata survives the round trip through the encoded schema.
        let again = logical_schema(&encoded_schema(&schema)).unwrap();
        assert_eq!(nd_logical_metadata(again.field(0)), Some(metadata));
        let plain = Schema::new(vec![nd_encoded_field_of(&Field::new(
            "x",
            DataType::Int32,
            true,
        ))]);
        assert!(nd_logical_metadata(logical_schema(&plain).unwrap().field(0)).is_none());
    }

    #[test]
    fn a_schema_names_its_coordinates_by_convention() {
        let schema = planned(
            &[
                ("time", &["time"]),
                ("lat", &["lat"]),
                ("sst", &["time", "lat", "lon"]),
                ("lon", &["lon", "time"]),
            ],
            |_| None,
        );
        let expected = vec![
            ("time".to_string(), "time".to_string()),
            ("lat".to_string(), "lat".to_string()),
        ];
        assert_eq!(schema_coordinates(&schema), expected);
    }

    #[test]
    fn axis_entries_override_the_convention_in_a_schema() {
        let schema = planned(
            &[
                ("t", &["time"]),
                ("time", &["time"]),
                ("N_PROF", &["N_PROF"]),
            ],
            |axis| match axis {
                "time" => Some(crate::AxisMeta::coordinate(
                    "t",
                    crate::AxisOrder::Ascending,
                )),
                _ => Some(crate::AxisMeta::no_coordinate()),
            },
        );
        let expected = vec![("time".to_string(), "t".to_string())];
        assert_eq!(schema_coordinates(&schema), expected);
    }
}
