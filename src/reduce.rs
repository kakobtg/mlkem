//! Modular-reduction helpers.
//!
//! Every function here may be called on secret data (secret-key
//! coefficients, the decrypted message, re-encryption noise), so all of
//! them are written to be constant-time: no data-dependent branches, no
//! data-dependent memory indexing, and no hardware division (`/` or `%` on
//! a secret operand compiles to `DIV`/`SDIV` on many targets, whose latency
//! depends on the operands — see KyberSlash, CVE-2023-51437). Division by
//! `q` is replaced by multiplication with a precomputed reciprocal and a
//! shift, and conditional corrections are done with sign masks.

use crate::params::MlKem768;

const Q: i32 = MlKem768::Q as i32;

/// Returns `x + q` if `x < 0`, else `x`. Branchless.
#[inline(always)]
pub fn caddq(x: i32) -> i32 {
    x + ((x >> 31) & Q)
}

/// Returns `x - q` if `x >= q`, else `x`. Branchless; valid for `x < q + 2^31`.
#[inline(always)]
pub fn csubq(x: i32) -> i32 {
    caddq(x - Q)
}

/// Normalize any `i32` to its representative in `[0, q)`, in constant time.
///
/// Approximates `floor(x / q)` as `(x * V) >> 38` with `V = round(2^38 / q)`.
/// For `|x| < 2^31` the approximation error is below `2^-8`, so the
/// estimated quotient is off by at most one and `x - q·est` lies in
/// `[-q, 2q)`; one conditional add and one conditional subtract finish it.
#[inline(always)]
pub fn mod_q(x: i32) -> i16 {
    const SHIFT: u32 = 38;
    const V: i64 = ((1i64 << SHIFT) + (Q as i64 / 2)) / Q as i64;
    // Arithmetic shift floors toward -inf, which is what we want for negatives.
    let est = ((x as i64) * V) >> SHIFT;
    let r = (x as i64 - est * Q as i64) as i32; // in [-q, 2q)
    csubq(caddq(r)) as i16
}

/// Constant-time `floor(n / q)` for `0 <= n < 2^24`.
///
/// Uses `M = ceil(2^48 / q)`; for `n < 2^24` the over-estimate
/// `n·(M − 2^48/q)/2^48 < 2^-24 < 1/q` never crosses an integer boundary, so
/// the result is exact (also checked exhaustively in the tests below).
#[inline(always)]
pub fn div_q(n: u32) -> u32 {
    const SHIFT: u32 = 48;
    const M: u64 = ((1u64 << SHIFT) + (Q as u64 - 1)) / Q as u64;
    debug_assert!(n < (1 << 24));
    ((n as u64 * M) >> SHIFT) as u32
}

/// Barrett reduction (typical)
pub fn barrett_reduce(x: i32) -> i16 {
    // v = floor(2^26 / Q), pre-rounded to minimize error
    const V: i32 = ((1 << 26) + (Q / 2)) / Q;

    // Approximate quotient and subtract; good for |x| < 2^15
    let t = ((V.wrapping_mul(x).wrapping_add(1 << 25)) >> 26).wrapping_mul(Q);
    (x - t) as i16
}

/// Montgomery reduction (optional alternative)
#[allow(dead_code)]
pub fn montgomery_reduce(x: i32) -> i16 {
    const QINV: i32 = 62209; // -q^{-1} mod 2^16

    // Computes (x * R^{-1}) mod Q with R = 2^16
    let u = (x.wrapping_mul(QINV)) & 0xFFFF;
    let t = (x + u * Q) >> 16;
    csubq(t) as i16
}

#[allow(dead_code)]
#[inline]
pub fn add(a: i16, b: i16) -> i16 {
    csubq(a as i32 + b as i32) as i16
}

#[inline]
pub fn sub(a: i16, b: i16) -> i16 {
    caddq(a as i32 - b as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mod_q_matches_rem_euclid() {
        let check = |x: i32| {
            assert_eq!(mod_q(x) as i32, x.rem_euclid(Q), "mod_q({x})");
        };
        for x in -200_000..200_000 {
            check(x);
        }
        for &x in &[i32::MIN, i32::MIN + 1, i32::MAX, i32::MAX - 1, -Q, Q, 2 * Q - 1] {
            check(x);
        }
        // Sparse sweep over the full range.
        let mut x: i64 = i32::MIN as i64;
        while x <= i32::MAX as i64 {
            check(x as i32);
            x += 65_537;
        }
    }

    #[test]
    fn div_q_is_exact() {
        for n in 0u32..(1 << 24) {
            assert_eq!(div_q(n), n / Q as u32, "div_q({n})");
        }
    }

    #[test]
    fn add_sub_canonical() {
        for a in 0..Q as i16 {
            for b in (0..Q as i16).step_by(7) {
                assert_eq!(add(a, b) as i32, (a as i32 + b as i32) % Q);
                assert_eq!(sub(a, b) as i32, (a as i32 - b as i32).rem_euclid(Q));
            }
        }
    }
}
