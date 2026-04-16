use crate::params::MlKem768;
use crate::reduce;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Poly(pub [i16; MlKem768::N]);

impl Poly {
    pub fn zero() -> Self { Poly([0i16; MlKem768::N]) }

    pub fn add(&self, rhs: &Poly) -> Poly {
        let mut out = [0i16; MlKem768::N];
        for i in 0..MlKem768::N {
            // Essential for bit-level consistency in linearity tests
            out[i] = reduce::barrett_reduce(self.0[i] as i32 + rhs.0[i] as i32);
        }
        Poly(out)
    }

    pub fn sub(&self, rhs: &Poly) -> Poly {
        let mut out = [0i16; MlKem768::N];
        for i in 0..MlKem768::N {
            out[i] = reduce::sub(self.0[i], rhs.0[i]);
        }
        Poly(out)
    }

    /// Schoolbook multiplication modulo (X^256 + 1).
    /// Fixed: Uses i64 for accumulation to prevent overflow during summing.
    pub fn mul_schoolbook_debug(&self, rhs: &Poly) -> Poly {
        let mut acc = [0i64; MlKem768::N];

        for i in 0..MlKem768::N {
            for j in 0..MlKem768::N {
                let prod = self.0[i] as i64 * rhs.0[j] as i64;
                let idx = i + j;
                if idx < MlKem768::N {
                    acc[idx] += prod;
                } else {
                    // Polynomial reduction rule: X^n = -1
                    acc[idx - MlKem768::N] -= prod;
                }
            }
        }

        let mut out = [0i16; MlKem768::N];
        for (i, val) in acc.iter().enumerate() {
            // Final reduction to field elements
            out[i] = reduce::mod_q((*val % MlKem768::Q as i64) as i32);
        }

        Poly(out)
    }
}

/// Vectors of polynomials (length k).
pub type PolyVec = [Poly; MlKem768::K];

#[allow(dead_code)]
pub fn polyvec_zero() -> PolyVec {
    // Fixed: Uses constant K instead of hardcoded length for flexibility.
    [Poly::zero(); MlKem768::K]
}