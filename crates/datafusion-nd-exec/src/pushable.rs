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
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::config::ConfigOptions;
    use datafusion::functions::math::{pi, random};
    use datafusion::logical_expr::{Operator, ScalarUDF};
    use datafusion::physical_expr::ScalarFunctionExpr;
    use datafusion::physical_expr::expressions::{binary, cast, col, in_list, lit};

    use super::*;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("lat", DataType::Int32, true),
            Field::new("sst", DataType::Float64, true),
        ])
    }

    /// A call of the function `udf` without arguments.
    fn call(udf: Arc<ScalarUDF>) -> Arc<dyn PhysicalExpr> {
        let config = Arc::new(ConfigOptions::default());
        Arc::new(ScalarFunctionExpr::try_new(udf, vec![], &schema(), config).unwrap())
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
        assert!(is_pushable_expr(&cast(lat, &s, DataType::Float64).unwrap()));
        assert!(is_pushable_expr(&lit(7i32)));
    }

    #[test]
    fn a_volatile_function_is_not_pushable() {
        let s = schema();
        let random = call(random());
        assert!(!is_pushable_expr(&random));
        // A whitelisted node over it is not pushable either.
        let wrapped = binary(random, Operator::Plus, lit(1.0f64), &s).unwrap();
        assert!(!is_pushable_expr(&wrapped));
    }

    #[test]
    fn an_immutable_function_is_pushable() {
        let s = schema();
        let pi = call(pi());
        assert!(is_pushable_expr(&pi));
        let wrapped = binary(pi, Operator::Plus, lit(1.0f64), &s).unwrap();
        assert!(is_pushable_expr(&wrapped));
    }

    /// The whitelist is closed: an expression type that it does not list stays
    /// above the broadcast.
    #[test]
    fn unlisted_expression_types_are_not_pushable() {
        let s = schema();
        let lat = col("lat", &s).unwrap();
        let in_list = in_list(lat, vec![lit(1i32), lit(2i32)], &false, &s).unwrap();
        assert!(!is_pushable_expr(&in_list));
        // A whitelisted parent does not make an unlisted child pushable.
        let wrapped = binary(in_list, Operator::And, lit(true), &s).unwrap();
        assert!(!is_pushable_expr(&wrapped));
    }
}
