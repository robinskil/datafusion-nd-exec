//! A crate outside `datafusion-nd-exec` adds its own nd node through the
//! registry.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use datafusion_nd_exec::array::encoding::encode_nd_record_batch;
use datafusion_nd_exec::array::{Dimension, Dimensions, NdArrowArray, NdRecordBatch};
use datafusion_nd_exec::exec::{
    NdBroadcastExec, NdExecutionPlan, NdSourceExec, SendableNdBatchStream,
};
use datafusion_nd_exec::{NdNodeRegistry, NdSinker, Sunk, probe_for};
use futures::TryStreamExt;

/// An nd node of another crate: it passes nd batches through unchanged.
#[derive(Debug, Clone)]
struct NdTagExec {
    input: Arc<dyn ExecutionPlan>,
    nd_input: Arc<dyn NdExecutionPlan>,
    registry: Arc<NdNodeRegistry>,
}

impl NdTagExec {
    fn try_new(input: Arc<dyn ExecutionPlan>, registry: Arc<NdNodeRegistry>) -> Result<Self> {
        let nd_input = registry
            .as_nd_plan(&input)
            .ok_or_else(|| DataFusionError::Plan("NdTagExec needs an nd input".into()))?;
        Ok(Self {
            input,
            nd_input,
            registry,
        })
    }
}

impl DisplayAs for NdTagExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NdTagExec")
    }
}

impl ExecutionPlan for NdTagExec {
    fn name(&self) -> &str {
        "NdTagExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::try_new(
            children.remove(0),
            self.registry.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        NdBroadcastExec::try_new(Arc::new(self.clone()), self.registry.clone())?
            .execute(partition, context)
    }
}

impl NdExecutionPlan for NdTagExec {
    fn execute_nd(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableNdBatchStream> {
        self.nd_input.execute_nd(partition, context)
    }
}

fn dims(spec: &[(&str, usize)]) -> Dimensions {
    Dimensions::try_new(
        spec.iter()
            .map(|(name, size)| Dimension::new(*name, *size))
            .collect(),
    )
    .unwrap()
}

fn source() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("lat", DataType::Int32, true),
        Field::new("sst", DataType::Float64, true),
    ]));
    let lat = NdArrowArray::try_new(
        Arc::new(Int32Array::from(vec![10, 20])),
        dims(&[("lat", 2)]),
    )
    .unwrap();
    let sst = NdArrowArray::try_new(
        Arc::new(Float64Array::from(vec![0.0, 0.1, 1.0, 1.1])),
        dims(&[("time", 2), ("lat", 2)]),
    )
    .unwrap();
    let nd =
        NdRecordBatch::try_new(schema, vec![lat, sst], dims(&[("time", 2), ("lat", 2)])).unwrap();
    let encoded = encode_nd_record_batch(&nd).unwrap();
    let schema = encoded.schema();
    let memory = MemorySourceConfig::try_new_exec(&[vec![encoded]], schema, None).unwrap();
    Arc::new(NdSourceExec::try_new(memory).unwrap())
}

#[tokio::test]
async fn a_probe_from_another_crate_makes_its_node_nd_aware() {
    let registry = Arc::new(NdNodeRegistry::new().with_probe(probe_for::<NdTagExec>()));
    let tag: Arc<dyn ExecutionPlan> =
        Arc::new(NdTagExec::try_new(source(), registry.clone()).unwrap());

    // The default registry does not know the node.
    assert!(NdBroadcastExec::try_new(tag.clone(), NdNodeRegistry::shared_default()).is_err());

    // The registry with the probe does.
    let broadcast = Arc::new(NdBroadcastExec::try_new(tag, registry).unwrap());
    let schema = broadcast.schema();
    let batches: Vec<_> = broadcast
        .execute(0, Arc::new(TaskContext::default()))
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let out = concat_batches(&schema, &batches).unwrap();
    assert_eq!(out.num_rows(), 4);
}

#[test]
fn the_session_registry_falls_back_to_the_default() {
    use datafusion::execution::config::SessionConfig;

    let plain = SessionConfig::new();
    let registry = NdNodeRegistry::from_session_config(&plain);
    assert!(registry.as_nd_plan(&source()).is_some());

    let custom = Arc::new(NdNodeRegistry::new().with_probe(probe_for::<NdTagExec>()));
    let config = SessionConfig::new().with_extension(custom.clone());
    assert!(Arc::ptr_eq(
        &NdNodeRegistry::from_session_config(&config),
        &custom
    ));
}

/// A flat node of another crate: it passes flat batches through unchanged.
#[derive(Debug)]
struct FlatTagExec {
    input: Arc<dyn ExecutionPlan>,
}

impl DisplayAs for FlatTagExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FlatTagExec")
    }
}

impl ExecutionPlan for FlatTagExec {
    fn name(&self) -> &str {
        "FlatTagExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            input: children.remove(0),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.input.execute(partition, context)
    }
}

/// The sink check of the other crate: a `FlatTagExec` becomes an `NdTagExec`.
#[derive(Debug)]
struct TagSinker;

impl NdSinker for TagSinker {
    fn try_sink(
        &self,
        parent: &Arc<dyn ExecutionPlan>,
        children: &[Arc<dyn ExecutionPlan>],
        registry: &Arc<NdNodeRegistry>,
    ) -> Result<Option<Sunk>> {
        let (true, [child]) = (parent.as_any().is::<FlatTagExec>(), children) else {
            return Ok(None);
        };
        let nd = NdTagExec::try_new(child.clone(), registry.clone())?;
        Ok(Some(Sunk::Below {
            nd: Arc::new(nd),
            residual: None,
        }))
    }
}

#[tokio::test]
async fn the_boundary_rule_sinks_a_node_of_another_crate() {
    use datafusion::common::config::ConfigOptions;
    use datafusion::physical_optimizer::PhysicalOptimizerRule;
    use datafusion::physical_plan::displayable;
    use datafusion_nd_exec::NdBoundaryRule;

    let plan = || -> Arc<dyn ExecutionPlan> {
        Arc::new(FlatTagExec {
            input: Arc::new(
                NdBroadcastExec::try_new(source(), NdNodeRegistry::shared_default()).unwrap(),
            ),
        })
    };

    // The default registry has no sinker for the node.
    let unchanged = NdBoundaryRule::new(NdNodeRegistry::shared_default())
        .optimize(plan(), &ConfigOptions::default())
        .unwrap();
    assert_eq!(unchanged.name(), "FlatTagExec");

    let registry = Arc::new(
        NdNodeRegistry::new()
            .with_probe(probe_for::<NdTagExec>())
            .with_sinker(Arc::new(TagSinker)),
    );
    let optimized = NdBoundaryRule::new(registry)
        .optimize(plan(), &ConfigOptions::default())
        .unwrap();
    let rendered = displayable(optimized.as_ref()).indent(true).to_string();
    assert!(
        rendered.starts_with("NdBroadcastExec: region=[NdTagExec, NdSourceExec]"),
        "{rendered}"
    );

    let schema = optimized.schema();
    let batches: Vec<_> = optimized
        .execute(0, Arc::new(TaskContext::default()))
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert_eq!(concat_batches(&schema, &batches).unwrap().num_rows(), 4);
}
