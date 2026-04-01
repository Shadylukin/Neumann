// SPDX-License-Identifier: MIT OR Apache-2.0
//! Riemannian gradient operations on the Poincare disk.
//!
//! Provides Euclidean gradient computation, Riemannian rescaling via the
//! inverse metric tensor, and full SGD update steps with automatic projection
//! back into the disk.
//!
//! All functions accept a `curvature` parameter matching the crate convention
//! where the Poincare metric is:
//!
//! ```text
//! g_ij = (4 / (c * (1 - ||x||^2)^2)) * delta_ij
//! ```

use crate::poincare::PoincarePoint;

/// Euclidean gradient of the Poincare distance `d_c(x, y)` with respect to `x`.
///
/// The distance formula is:
///
/// ```text
/// d_c(x,y) = arcosh(gamma) / sqrt(c)
///   where gamma = 1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2))
/// ```
///
/// Returns a zero vector when `x` and `y` are (nearly) identical.
#[must_use]
pub fn euclidean_grad_distance(x: &[f64], y: &[f64], curvature: f64) -> Vec<f64> {
    let dim = x.len();
    let x_norm_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let y_norm_sq: f64 = y.iter().map(|yi| yi * yi).sum();
    let alpha = 1.0 - x_norm_sq;
    let beta = 1.0 - y_norm_sq;
    let diff_sq: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi) * (xi - yi))
        .sum();

    let gamma = 1.0 + 2.0 * diff_sq / (alpha * beta);

    if gamma.mul_add(gamma, -1.0) < 1e-15 {
        return vec![0.0; dim];
    }

    let denom = gamma.mul_add(gamma, -1.0).sqrt() * curvature.sqrt();
    let alpha_sq_beta = alpha * alpha * beta;

    (0..dim)
        .map(|i| {
            let d_gamma = (4.0 / alpha_sq_beta) * (alpha * (x[i] - y[i]) + diff_sq * x[i]);
            d_gamma / denom
        })
        .collect()
}

/// Convert a Euclidean gradient to a Riemannian gradient.
///
/// Applies the inverse metric tensor of the Poincare disk:
///
/// ```text
/// g^{-1}_ij = c * (1 - ||x||^2)^2 / 4 * delta_ij
/// ```
#[must_use]
pub fn riemannian_grad(x_norm_sq: f64, euclidean_grad: &[f64], curvature: f64) -> Vec<f64> {
    let alpha = 1.0 - x_norm_sq;
    let scale = curvature * alpha * alpha / 4.0;
    euclidean_grad.iter().map(|&g| scale * g).collect()
}

/// Perform a full Riemannian SGD step.
///
/// Computes the Riemannian gradient from the Euclidean gradient, updates
/// the point, and projects back into the Poincare disk:
///
/// ```text
/// x_new = x - lr * (c * (1 - ||x||^2)^2 / 4) * euclidean_grad
/// ```
#[must_use]
pub fn riemannian_update(
    point: &PoincarePoint,
    euclidean_grad: &[f64],
    lr: f64,
    curvature: f64,
) -> PoincarePoint {
    let norm_sq = point.norm_sq();
    let alpha = 1.0 - norm_sq;
    let scale = curvature * alpha * alpha / 4.0;

    let new_coords: Vec<f64> = point
        .coords()
        .iter()
        .zip(euclidean_grad.iter())
        .map(|(&xi, &gi)| lr.mul_add(-scale * gi, xi))
        .collect();

    PoincarePoint::new(new_coords)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn riemannian_grad_scales_with_curvature() {
        let eg = vec![1.0, 0.0];
        let rg1 = riemannian_grad(0.25, &eg, 1.0);
        let rg2 = riemannian_grad(0.25, &eg, 2.0);
        for (a, b) in rg1.iter().zip(rg2.iter()) {
            assert!(
                (b - 2.0 * a).abs() < 1e-12,
                "c=2 should produce 2x the gradient of c=1"
            );
        }
    }

    #[test]
    fn gradient_at_origin_no_curvature_in_denom() {
        let eg = vec![1.0, 0.0];
        let rg = riemannian_grad(0.0, &eg, 2.0);
        // scale = c * 1^2 / 4 = 2/4 = 0.5
        assert!((rg[0] - 0.5).abs() < 1e-12);
        assert!(rg[1].abs() < 1e-12);
    }

    #[test]
    fn gradient_near_boundary_damped() {
        let eg = vec![1.0, 0.0];
        let norm_sq = 0.99 * 0.99;
        let rg = riemannian_grad(norm_sq, &eg, 1.0);
        let expected = (1.0 - norm_sq) * (1.0 - norm_sq) / 4.0;
        assert!((rg[0] - expected).abs() < 1e-10);
        assert!(rg[0] < 1e-3, "gradient near boundary should be damped");
    }

    #[test]
    fn gradient_pushes_apart() {
        let x = &[0.1, 0.0];
        let y = &[0.3, 0.0];
        let grad = euclidean_grad_distance(x, y, 1.0);
        // Moving x left increases distance from y, so gradient points left
        assert!(grad[0] < 0.0, "gradient should push x away from y");
    }

    #[test]
    fn gradient_zero_for_same_point() {
        let x = &[0.3, 0.4];
        let grad = euclidean_grad_distance(x, x, 1.0);
        for g in &grad {
            assert!(g.abs() < 1e-12, "gradient should be zero for same point");
        }
    }

    #[test]
    fn update_stays_in_disk() {
        let point = PoincarePoint::new(vec![0.8, 0.5]);
        let grad = vec![10.0, -10.0];
        let updated = riemannian_update(&point, &grad, 1.0, 1.0);
        assert!(updated.is_valid(), "updated point must remain in disk");
    }

    #[test]
    fn update_zero_grad_no_movement() {
        let point = PoincarePoint::new(vec![0.3, 0.4]);
        let grad = vec![0.0, 0.0];
        let updated = riemannian_update(&point, &grad, 0.1, 1.0);
        for (a, b) in updated.coords().iter().zip(point.coords().iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn update_reduces_distance() {
        let point = PoincarePoint::new(vec![0.5, 0.0]);
        let target = PoincarePoint::new(vec![0.1, 0.0]);
        let grad = euclidean_grad_distance(point.coords(), target.coords(), 1.0);
        let updated = riemannian_update(&point, &grad, 0.1, 1.0);
        let d_before = point.distance(&target, 1.0);
        let d_after = updated.distance(&target, 1.0);
        assert!(
            d_after < d_before,
            "step should reduce distance: {d_after} vs {d_before}"
        );
    }

    #[test]
    fn gradient_finite_difference() {
        let eps = 1e-7;
        for curvature in [0.5, 1.0, 2.0] {
            let x = vec![0.3, 0.2];
            let y = vec![-0.1, 0.4];
            let analytical = euclidean_grad_distance(&x, &y, curvature);

            let py = PoincarePoint::new(y.clone());

            for i in 0..2 {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[i] += eps;
                x_minus[i] -= eps;

                let p_plus = PoincarePoint::new(x_plus);
                let p_minus = PoincarePoint::new(x_minus);

                let d_plus = p_plus.distance(&py, curvature);
                let d_minus = p_minus.distance(&py, curvature);
                let numerical = (d_plus - d_minus) / (2.0 * eps);

                assert!(
                    (analytical[i] - numerical).abs() < 1e-5,
                    "finite difference mismatch at dim {i}, c={curvature}: \
                     analytical={}, numerical={}",
                    analytical[i],
                    numerical,
                );
            }
        }
    }
}
