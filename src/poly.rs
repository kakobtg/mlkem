use crate::params::MlKem768;
use crate::reduce;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Poly(pub [i16; MlKem768::N]);

impl Poly {
    pub fn zero() -> Self {
        Poly([0i16; MlKem768::N])
    }

    pub fn add(&self, rhs: &Poly) -> Poly {
        let mut out = [0i16; MlKem768::N];
        for i in 0..MlKem768::N {
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

    /// Reduces every coefficient to its canonical representative in [0, q).
    pub fn reduced(mut self) -> Poly {
        for coef in self.0.iter_mut() {
            *coef = reduce::mod_q(*coef as i32);
        }
        self
    }

    /// Schoolbook multiplication modulo (X^256 + 1). Accumulates in i64 to
    /// avoid overflow.
    pub fn mul_schoolbook_debug(&self, rhs: &Poly) -> Poly {
        let mut acc = [0i64; MlKem768::N];

        for i in 0..MlKem768::N {
            for j in 0..MlKem768::N {
                let prod = self.0[i] as i64 * rhs.0[j] as i64;
                let idx = i + j;
                if idx < MlKem768::N {
                    acc[idx] += prod;
                } else {
                    // X^n = -1
                    acc[idx - MlKem768::N] -= prod;
                }
            }
        }

        let mut out = [0i16; MlKem768::N];
        for (i, val) in acc.iter().enumerate() {
            out[i] = reduce::mod_q((*val % MlKem768::Q as i64) as i32);
        }

        Poly(out)
    }
}

/// Vectors of polynomials (length k).
pub type PolyVec = [Poly; MlKem768::K];

#[allow(dead_code)]
pub fn polyvec_zero() -> PolyVec {
    [Poly::zero(); MlKem768::K]
}
