//! Which expressions can run below the nd boundary.

use std::sync::Arc;

use datafusion::logical_expr::Volatility;
use datafusion::physical_expr::expressions::{
    BinaryExpr, CaseExpr, CastExpr, Column, IsNotNullExpr, IsNullExpr, Literal, NegativeExpr,
    NotExpr, TryCastExpr,
};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};

/// Whether an expression can be evaluated before broadcast and give the same
/// result after broadcast — i.e. it is element-wise and deterministic.
///
/// Conservative: an expression built only from a whitelist of element-wise node
/// types (arithmetic/comparison/boolean, cast, negation, null checks, `CASE`)
/// and non-volatile scalar functions is pushable. Any node outside the
/// whitelist — a volatile function like `random()`, a window/subquery
/// expression, or anything unrecognized — makes the whole expression stay above
/// the broadcast.
pub fn is_pushable_expr(expr: &Arc<dyn PhysicalExpr>) -> bool {
    is_elementwise_node(expr) && expr.children().iter().all(|c| is_pushable_expr(c))
}

/// Whether a single node (ignoring its children) is a known element-wise,
/// deterministic operator.
fn is_elementwise_node(expr: &Arc<dyn PhysicalExpr>) -> bool {
    let any = expr.as_any();
    if any.is::<Column>()
        || any.is::<Literal>()
        || any.is::<BinaryExpr>()
        || any.is::<CastExpr>()
        || any.is::<TryCastExpr>()
        || any.is::<NegativeExpr>()
        || any.is::<NotExpr>()
        || any.is::<IsNullExpr>()
        || any.is::<IsNotNullExpr>()
        || any.is::<CaseExpr>()
    {
        return true;
    }
    // Scalar functions are row-wise, but a volatile one (e.g. `random()`) would
    // produce fewer distinct values if evaluated before broadcast.
    if let Some(func) = any.downcast_ref::<ScalarFunctionExpr>() {
        return func.fun().signature().volatility != Volatility::Volatile;
    }
    false
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::config::ConfigOptions;
    use datafusion::error::Result;
    use datafusion::logical_expr::{
        ColumnarValue, Operator, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
        Volatility,
    };
    use datafusion::physical_expr::ScalarFunctionExpr;
    use datafusion::physical_expr::expressions::{binary, cast, col, lit};
    use datafusion::scalar::ScalarValue;

    use super::*;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("lat", DataType::Int32, true),
            Field::new("sst", DataType::Float64, true),
        ])
    }

    #[test]
    fn arithmetic_cast_literal_are_pushable() {
        let s = schema();
        let lat = col("lat", &s).unwrap();
        // (lat * 2) + 1
        let arith = binary(
            binary(lat.clone(), Operator::Multiply, lit(2i32), &s).unwrap(),
            Operator::Plus,
            lit(1i32),
            &s,
        )
        .unwrap();
        assert!(is_pushable_expr(&arith));

        // CAST(lat AS Float64)
        let casted = cast(lat, &s, DataType::Float64).unwrap();
        assert!(is_pushable_expr(&casted));

        // a bare literal
        let seven: Arc<dyn PhysicalExpr> = lit(7i32);
        assert!(is_pushable_expr(&seven));
    }

    /// A minimal volatile scalar function, to prove the volatility guard.
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct VolatileUdf {
        signature: Signature,
    }

    impl ScalarUDFImpl for VolatileUdf {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn name(&self) -> &str {
            "test_volatile"
        }
        fn signature(&self) -> &Signature {
            &self.signature
        }
        fn return_type(&self, _: &[DataType]) -> Result<DataType> {
            Ok(DataType::Float64)
        }
        fn invoke_with_args(&self, _: ScalarFunctionArgs) -> Result<ColumnarValue> {
            Ok(ColumnarValue::Scalar(ScalarValue::Float64(Some(0.0))))
        }
    }

    #[test]
    fn volatile_scalar_function_is_not_pushable() {
        let s = schema();
        let udf = Arc::new(ScalarUDF::new_from_impl(VolatileUdf {
            signature: Signature::exact(vec![], Volatility::Volatile),
        }));
        let func = ScalarFunctionExpr::try_new(udf, vec![], &s, Arc::new(ConfigOptions::default()))
            .unwrap();
        let expr: Arc<dyn PhysicalExpr> = Arc::new(func);
        assert!(!is_pushable_expr(&expr));

        // …and a whitelisted node wrapping it is tainted too.
        let wrapped = binary(expr, Operator::Plus, lit(1.0f64), &s).unwrap();
        assert!(!is_pushable_expr(&wrapped));
    }

    /// The mirror of the volatility guard: an immutable scalar function is
    /// element-wise, so it may be evaluated on the footprint sub-grid.
    #[test]
    fn immutable_scalar_function_is_pushable() {
        #[derive(Debug, PartialEq, Eq, Hash)]
        struct ImmutableUdf {
            signature: Signature,
        }

        impl ScalarUDFImpl for ImmutableUdf {
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn name(&self) -> &str {
                "test_immutable"
            }
            fn signature(&self) -> &Signature {
                &self.signature
            }
            fn return_type(&self, _: &[DataType]) -> Result<DataType> {
                Ok(DataType::Float64)
            }
            fn invoke_with_args(&self, _: ScalarFunctionArgs) -> Result<ColumnarValue> {
                Ok(ColumnarValue::Scalar(ScalarValue::Float64(Some(0.0))))
            }
        }

        let s = schema();
        let udf = Arc::new(ScalarUDF::new_from_impl(ImmutableUdf {
            signature: Signature::exact(vec![], Volatility::Immutable),
        }));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(
            ScalarFunctionExpr::try_new(udf, vec![], &s, Arc::new(ConfigOptions::default()))
                .unwrap(),
        );
        assert!(is_pushable_expr(&expr));

        // Wrapping it in whitelisted nodes keeps it pushable.
        let wrapped = binary(expr, Operator::Plus, lit(1.0f64), &s).unwrap();
        assert!(is_pushable_expr(&wrapped));
    }

    /// The whitelist is closed: an element-wise expression type that simply is
    /// not listed stays above the broadcast rather than being assumed safe.
    #[test]
    fn unlisted_expression_types_are_not_pushable() {
        use datafusion::physical_expr::expressions::in_list;

        let s = schema();
        let in_list = in_list(
            col("lat", &s).unwrap(),
            vec![lit(1i32), lit(2i32)],
            &false,
            &s,
        )
        .unwrap();
        assert!(!is_pushable_expr(&in_list));

        // A whitelisted parent does not launder an unlisted child.
        let wrapped = binary(in_list, Operator::And, lit(true), &s).unwrap();
        assert!(!is_pushable_expr(&wrapped));
    }
}
