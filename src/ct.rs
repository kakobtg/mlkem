//! Constant-time helpers built on `subtle`, whose `Choice` type sits behind an
//! optimisation barrier so the compiler cannot turn the masks back into
//! branches.

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

pub fn ct_eq(a: &[u8], b: &[u8]) -> Choice {
    a.ct_eq(b)
}

#[allow(dead_code)]
pub fn ct_select_u8(choice: Choice, a: u8, b: u8) -> u8 {
    // `conditional_select(x, y, c)` returns `y` when `c == 1`.
    u8::conditional_select(&b, &a, choice)
}

/// Returns `a` if `choice == 1`, else `b`, without branching on `choice`.
pub fn ct_select_bytes<const N: usize>(choice: Choice, a: &[u8; N], b: &[u8; N]) -> [u8; N] {
    let mut out = [0u8; N];
    for i in 0..N {
        out[i] = u8::conditional_select(&b[i], &a[i], choice);
    }
    out
}
