use subtle::{Choice, ConstantTimeEq};

pub fn ct_eq(a: &[u8], b: &[u8]) -> Choice {
    a.ct_eq(b)
}

#[allow(dead_code)]
pub fn ct_select_u8(choice: Choice, a: u8, b: u8) -> u8 {
    let mask = (0u8).wrapping_sub(choice.unwrap_u8());
    (a & mask) | (b & !mask)
}

pub fn ct_select_bytes<const N: usize>(choice: Choice, a: &[u8; N], b: &[u8; N]) -> [u8; N] {
    let mut out = [0u8; N];
    let mask = (0u8).wrapping_sub(choice.unwrap_u8());

    for i in 0..N {
        out[i] = b[i] ^ (mask & (a[i] ^ b[i]));
    }
    out
}
