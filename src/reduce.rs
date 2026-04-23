use crate::params::MlKem768;

/// Normalize to representative in [0, q)
pub fn mod_q(x: i32) -> i16 {
    const Q: i32 = MlKem768::Q as i32;
    let mut r = x % Q;
    if r < 0 { r += Q; }
    r as i16
}

/// Barrett reduction (typical)
pub fn barrett_reduce(x: i32) -> i16 {
    const Q: i32 = 3329;
    // v = floor(2^26 / Q), pre-rounded to minimize error
    const V: i32 = ((1 << 26) + (Q / 2)) / Q;

    // Approximate quotient and subtract; good for |x| < 2^15
    let t = ((V.wrapping_mul(x).wrapping_add(1 << 25)) >> 26).wrapping_mul(Q);
    (x - t) as i16
}

/// Montgomery reduction (optional alternative)
#[allow(dead_code)]
pub fn montgomery_reduce(x: i32) -> i16 {
    const Q: i32 = 3329;
    const QINV: i32 = 62209; // -q^{-1} mod 2^16

    // Computes (x * R^{-1}) mod Q with R = 2^16
    let u = (x.wrapping_mul(QINV)) & 0xFFFF;
    let mut t = (x + u * Q) >> 16;
    if t >= Q { t -= Q; }
    t as i16
}

#[allow(dead_code)]
#[allow(dead_code)]
#[inline]
pub fn add(a: i16, b: i16) -> i16 {
    let mut r = a as i32 + b as i32;
    if r >= MlKem768::Q as i32 {
        r -= MlKem768::Q as i32;
    }
    r as i16
}

#[inline]
pub fn sub(a: i16, b: i16) -> i16 {
    let mut r = a as i32 - b as i32;
    if r < 0 {
        r += MlKem768::Q as i32;
    }
    r as i16
}
