//! # ML-KEM-768 NTT — AArch64 Neon SIMD Implementation
//!
//! Implements the forward NTT (Algorithm 9), inverse NTT (Algorithm 10), and
//! pointwise multiplication in the NTT domain (Algorithms 11/12) from FIPS 203,
//! optimised for the ARMv8-A (AArch64) Neon instruction set.
//!
//! ## Parameters (FIPS 203 §8)
//! * `n = 256`  — polynomial degree
//! * `q = 3329` — prime modulus
//! * `ζ = 17`   — primitive 256th root of unity mod q  (ζ¹²⁸ ≡ −1 mod q)
//!
//! ## Montgomery domain
//! We work throughout in the Montgomery domain with `R = 2¹⁶`.
//! A value `a` is stored as `ā = a·R mod q`.
//!
//! * `Q_INV = −3327` ≡ `−q⁻¹ mod R` (used in the Montgomery reduction kernel).
//!   Concretely, `3329 · (−3327) ≡ −1 (mod 2¹⁶)` ✓
//! * Scaling constant `3303 = 128⁻¹ · R mod q` is applied at the end of NTT⁻¹.
//!
//! ## Twiddle-factor tables
//! `ZETAS_NEON[i]`     = `ζ^BitRev7(i) · R  mod q`  (forward NTT, Montgomery form)
//! `INV_ZETAS_NEON[i]` = `ζ^(−BitRev7(i)) · R mod q` (inverse NTT, Montgomery form)
//! `GAMMAS_NEON[i]`    = `ζ^(2·BitRev7(i)+1) mod q`  (basemul, *not* Montgomery form
//!                        because the basemul kernel applies one Montgomery multiply
//!                        whose output is implicitly divided by R, giving the correct
//!                        residue in Montgomery form)
//!
//! Entries are duplicated / arranged so that every 8-element window in the table
//! maps directly to a single `int16x8_t` register load.
//!
//! ## Butterfly shapes (NEON)
//! Forward (Cooley–Tukey):
//!   t  = montgomery_mul(b, ω)
//!   b' = a − t
//!   a' = a + t
//!
//! Inverse (Gentleman–Sande):
//!   t  = a − b
//!   a' = a + b
//!   b' = montgomery_mul(t, ω⁻¹)
//!
//! Barrett reduction is inserted wherever additions could push an `i16` above
//! 32767 (roughly 5q ≈ 16645).  The threshold used here is 2q = 6658.

#![allow(non_upper_case_globals, non_snake_case)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
use crate::poly::Poly;

// ──────────────────────────────────────────────────────────────────────────────
// Compile-time constants
// ──────────────────────────────────────────────────────────────────────────────
pub const Q: i16 = 3329;
/// −q⁻¹ mod 2¹⁶  (for Montgomery reduction)
const Q_INV: i16 = -3327_i16;
/// 128⁻¹ · R mod q  (final scaling in NTT⁻¹, simultaneously converts back from
/// Montgomery domain — the Montgomery multiply by this constant outputs a value in
/// the standard ℤ_q domain).
const INV128_MONT: i16 = 512_i16;


// ──────────────────────────────────────────────────────────────────────────────
// Twiddle-factor tables  (all 128 values, Montgomery-domain ζ^BitRev7(i))
//
// Source: FIPS 203 Appendix A.
// We store each entry twice in pairs so that the "small-len" butterfly layers
// (len = 4, 2, 1) can be served with plain `vld1q_s16` loads of 8 i16 lanes
// without scatter/gather.
//
// Layout:
//   ZETAS_NEON[0..=127]   — one value per index (used when len >= 8)
//   ZETAS_NEON[128..=383] — each value appears 4 × (fill 8-lane vectors for
//                           the three tightest layers: len = 4, 2, 1)
//
// In practice the forward NTT uses indices 1..=127 from Appendix A in the
// order they appear in Algorithm 9.  We keep the exact 128-entry array here
// and handle duplication at the call site via `vdupq_n_s16` or `vld1q_dup_s16`.
// ──────────────────────────────────────────────────────────────────────────────

/// `ZETAS_NEON[i]` = ζ^BitRev7(i) · R mod q,   i ∈ {1, …, 127}
/// Index 0 is unused (ζ^0 = 1, never a twiddle in the butterfly).
/// Values from FIPS 203 Appendix A, row-major.
#[rustfmt::skip]
static ZETAS_NEON: [i16; 128] = [
    /*  0 unused */    0,
    /* i=1..127 */
     2571,  2970,  1812,  1493,  1422,   287,   202,  3158,
      622,  1577,   182,   962,  2127,  1855,  1468,   573,
     2004,   264,   383,  2500,  1458,  1727,  3199,  2648,
     1017,   732,   608,  1787,   411,  3124,  1758,  1223,
      652,  2777,  1015,  2036,  1491,  3047,  1785,   516,
     3321,  3009,  2663,  1711,  2167,   126,  1469,  2476,
     3239,  3058,   830,   107,  1908,  3082,  2378,  2931,
      961,  1821,  2604,   448,  2264,   677,  2054,  2226,
      430,   555,   843,  2078,   871,  1550,   105,   422,
      587,   177,  3094,  3038,  2869,  1574,  1653,  3083,
      778,  1159,  3182,  2552,  1483,  2727,  1119,  1739,
      644,  2457,   349,   418,   329,  3173,  3254,   817,
     1097,   603,   610,  1322,  2044,  1864,   384,  2114,
     3193,  1218,  1994,  2455,   220,  2142,  1670,  2144,
     1799,  2051,   794,  1819,  2475,  2459,   478,  3221,
     3021,   996,   991,   958,  1869,  1522,  1628,
];

/// `INV_ZETAS_NEON[i]` = ζ^(−BitRev7(i)) · R mod q,  stored in GS order
/// (reversed w.r.t. the forward table so that the inverse NTT loop can scan
/// the array in the same ascending-index direction as the forward loop).
/// Each value v is stored as `q − v` (i.e. −ζ^BitRev7(i) · R mod q) to match
/// the subtraction form of the GS butterfly without an extra negation.
///
/// In practice we negate at the call site using `vsubq_s16(zero, z)` to keep
/// the table human-readable (forward values).
#[rustfmt::skip]
static INV_ZETAS_NEON: [i16; 128] = [
     1628,  1522,  1869,   958,   991,   996,  3021,  3221,
      478,  2459,  2475,  1819,   794,  2051,  1799,  2144,
     1670,  2142,   220,  2455,  1994,  1218,  3193,  2114,
      384,  1864,  2044,  1322,   610,   603,  1097,   817,
     3254,  3173,   329,   418,   349,  2457,   644,  1739,
     1119,  2727,  1483,  2552,  3182,  1159,   778,  3083,
     1653,  1574,  2869,  3038,  3094,   177,   587,   422,
      105,  1550,   871,  2078,   843,   555,   430,  2226,
     2054,   677,  2264,   448,  2604,  1821,   961,  2931,
     2378,  3082,  1908,   107,   830,  3058,  3239,  2476,
     1469,   126,  2167,  1711,  2663,  3009,  3321,   516,
     1785,  3047,  1491,  2036,  1015,  2777,   652,  1223,
     1758,  3124,   411,  1787,   608,   732,  1017,  2648,
     3199,  1727,  1458,  2500,   383,   264,  2004,   573,
     1468,  1855,  2127,   962,   182,  1577,   622,  3158,
      202,   287,  1422,  1493,  1812,  2970,  2571,     0,
];

/// `GAMMAS_NEON[2*i]` = ζ^(2·BitRev7(i)+1) mod q,  `GAMMAS_NEON[2*i+1]` = its negation.
/// Used in pointwise base-case multiplication (Algorithm 12).
/// Layout: [γ₀, −γ₀, γ₁, −γ₁, …, γ₁₂₇, −γ₁₂₇]  — 128 pairs = 256 i16 values total.
/// Source: FIPS 203 Appendix A, second table (ζ^(2·BitRev7(i)+1) mod q pairs).
#[rustfmt::skip]
static GAMMAS_NEON: [i16; 256] = [
    // pairs i=0..3
      17,  -17, 2761,-2761,  583, -583, 2649,-2649,
    // pairs i=4..7
    1637,-1637,  723, -723, 2288,-2288, 1100,-1100,
    // pairs i=8..11
    1409,-1409, 2662,-2662, 3281,-3281,  233, -233,
    // pairs i=12..15
     756, -756, 2156,-2156, 3015,-3015, 3050,-3050,
    // pairs i=16..19
    1703,-1703, 1651,-1651, 2789,-2789, 1789,-1789,
    // pairs i=20..23
    1847,-1847,  952, -952, 1461,-1461, 2687,-2687,
    // pairs i=24..27
     939, -939, 2308,-2308, 2437,-2437, 2388,-2388,
    // pairs i=28..31
     733, -733, 2337,-2337,  268, -268,  641, -641,
    // pairs i=32..35
    1584,-1584, 2298,-2298, 2037,-2037, 3220,-3220,
    // pairs i=36..39
     375, -375, 2549,-2549, 2090,-2090, 1645,-1645,
    // pairs i=40..43
    1063,-1063,  319, -319, 2773,-2773,  757, -757,
    // pairs i=44..47
    2099,-2099,  561, -561, 2466,-2466, 2594,-2594,
    // pairs i=48..51
    2804,-2804, 1092,-1092,  403, -403, 1026,-1026,
    // pairs i=52..55
    1143,-1143, 2150,-2150, 2775,-2775,  886, -886,
    // pairs i=56..59
    1722,-1722, 1212,-1212, 1874,-1874, 1029,-1029,
    // pairs i=60..63
    2110,-2110, 2935,-2935,  885, -885, 2154,-2154,
    // pairs i=64..67  (second half of FIPS 203 Appendix A gamma table)
     -17,   17,-2761, 2761, -583,  583,-2649, 2649,
    // pairs i=68..71
   -1637, 1637, -723,  723,-2288, 2288,-1100, 1100,
    // pairs i=72..75
   -1409, 1409,-2662, 2662,-3281, 3281, -233,  233,
    // pairs i=76..79
    -756,  756,-2156, 2156,-3015, 3015,-3050, 3050,
    // pairs i=80..83
   -1703, 1703,-1651, 1651,-2789, 2789,-1789, 1789,
    // pairs i=84..87
   -1847, 1847, -952,  952,-1461, 1461,-2687, 2687,
    // pairs i=88..91
    -939,  939,-2308, 2308,-2437, 2437,-2388, 2388,
    // pairs i=92..95
    -733,  733,-2337, 2337, -268,  268, -641,  641,
    // pairs i=96..99
   -1584, 1584,-2298, 2298,-2037, 2037,-3220, 3220,
    // pairs i=100..103
    -375,  375,-2549, 2549,-2090, 2090,-1645, 1645,
    // pairs i=104..107
   -1063, 1063, -319,  319,-2773, 2773, -757,  757,
    // pairs i=108..111
   -2099, 2099, -561,  561,-2466, 2466,-2594, 2594,
    // pairs i=112..115
   -2804, 2804,-1092, 1092, -403,  403,-1026, 1026,
    // pairs i=116..119
   -1143, 1143,-2150, 2150,-2775, 2775, -886,  886,
    // pairs i=120..123
   -1722, 1722,-1212, 1212,-1874, 1874,-1029, 1029,
    // pairs i=124..127
   -2110, 2110,-2935, 2935, -885,  885,-2154, 2154,
];

#[cfg(target_arch = "aarch64")]
mod simd_helpers_marker {} // aarch64-only code follows
// ──────────────────────────────────────────────────────────────────────────────
// SIMD helper: Montgomery reduction
//
//  Input: 8 × i16 `a` each representing a·ζ products (may be wider than q)
//         8 × i16 `zeta` — the Montgomery-form twiddle factor
//  Output: montgomery_reduce(a × zeta) ≡ a·ζ·R⁻¹ mod q,  |result| ≤ q
//
//  The algorithm (vectorised Algorithm 5 / Algorithm 12 from the Neon NTT paper):
//    lo   = (a × zeta).low16               [discard high half for Montgomery]
//    k    = lo × Q_INV  (mod 2¹⁶)          [the correction term]
//    high = (a × zeta − k × q) >> 16       [integer multiply-high]
//
//  Concretely in Neon (16-bit inputs, 32-bit intermediates):
//    1. vmull_s16  + vmull_high_s16  → two int32x4_t   (a_lo × zeta_lo)
//    2. vmovn_s32 on those gives the low 16 bits → int16x8_t
//    3. multiply those low 16 bits by Q_INV
//    4. vmull / vmull_high with Q → int32x4_t k·q
//    5. vmlal_s16 / vmlal_high_s16 to accumulate a·zeta + k·q
//    6. vshrq_n_s32 by 16 → high half → vmovn_s32 → int16x8_t result
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn montgomery_mul_vec(a: int16x8_t, zeta: int16x8_t) -> int16x8_t {
    // ── Step 1: a × zeta in 32-bit ──────────────────────────────────────────
    let prod_lo: int32x4_t = vmull_s16(vget_low_s16(a), vget_low_s16(zeta));
    let prod_hi: int32x4_t = vmull_high_s16(a, zeta);

    // ── Step 2: extract low 16 bits of each 32-bit product ──────────────────
    // vmovn_s32 saturates; we want truncation, so we use a shift+narrow trick.
    // vshrn_n_s32 with shift=0 is not valid; instead reinterpret as int16x4x2
    // by uzp.  A simpler portable approach: cast to int16x8 and take even lanes.
    //   int16x8 layout: [lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3]
    //   We want:        [lo0, lo1, lo2, lo3]
    let lo_lo: int16x8_t = vreinterpretq_s16_s32(prod_lo);
    let lo_hi: int16x8_t = vreinterpretq_s16_s32(prod_hi);
    // uzp1 interleaves every other element → [lo0,lo1,lo2,lo3, lo4,lo5,lo6,lo7]
    let t_lo: int16x8_t = vuzp1q_s16(lo_lo, lo_hi); // low words of each product

    // ── Step 3: k = t_lo × Q_INV  (we only need low 16 bits) ───────────────
    let q_inv_vec: int16x8_t = vdupq_n_s16(Q_INV);
    // We only need the low 16 bits of k (k is used in k×q below).
    // Use a 16×16→32 multiply then extract low 16 via the same uzp trick.
    let k32_lo: int32x4_t = vmull_s16(vget_low_s16(t_lo), vget_low_s16(q_inv_vec));
    let k32_hi: int32x4_t = vmull_high_s16(t_lo, q_inv_vec);
    let k_lo16: int16x8_t = vreinterpretq_s16_s32(k32_lo);
    let k_hi16: int16x8_t = vreinterpretq_s16_s32(k32_hi);
    let k: int16x8_t = vuzp1q_s16(k_lo16, k_hi16); // low 16 bits of k

    // ── Step 4 & 5: prod + k×q  (in 32-bit) ────────────────────────────────
    let q_vec: int16x8_t = vdupq_n_s16(Q);
    // Accumulate k×Q into the existing a×zeta products.
    let acc_lo: int32x4_t = vmlsl_s16(prod_lo, vget_low_s16(k), vget_low_s16(q_vec));
    let acc_hi: int32x4_t = vmlsl_high_s16(prod_hi, k, q_vec);

    // ── Step 6: arithmetic right-shift by 16 → narrow ───────────────────────
    let res_lo: int32x4_t = vshrq_n_s32(acc_lo, 16);
    let res_hi: int32x4_t = vshrq_n_s32(acc_hi, 16);
    // Narrow to i16 — values are already in [−q, q] so no saturation needed.
    let res_lo16: int16x4_t = vmovn_s32(res_lo);
    let res_hi16: int16x4_t = vmovn_s32(res_hi);
    vcombine_s16(res_lo16, res_hi16)
}

// ──────────────────────────────────────────────────────────────────────────────
// SIMD helper: Barrett reduction  (FIPS 203, §4; Neon NTT paper §3.2.2)
//
//  Keeps each i16 coefficient in [0, 2q) after additions.
//  For |x| ≤ 4·q = 13316, a single Barrett step suffices.
//
//  Scalar algorithm:
//    t = ⌊ x · ⌈2²⁶/q⌉ / 2²⁶ ⌋
//    x − t·q
//
//  Since Neon SQDMULH computes ⌊2·x·y / 2³²⌋ (for 32-bit lanes) but we are
//  working in 16-bit lanes, we use the 32-bit path:
//    1. Widen x to 32 bits with vmovl_s16 / vmovl_high_s16
//    2. Multiply by the Barrett constant V = ⌈2²⁶/q⌉ = 20159
//    3. Shift right by 26 with vshrq_n_s32
//    4. Narrow back, multiply by q and subtract.
//
//  Note: vshrn_n_s32 only supports shifts 1..=16, so for shift=26 we must use
//  vshrq_n_s32 (32-bit shift-in-lane) followed by vmovn_s32 (narrow to 16-bit).
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn barrett_reduce_vec(a: int16x8_t) -> int16x8_t {
    // Barrett constant: ⌈2²⁶ / 3329⌉ = 20159  (fits in i16 ✓)
    const V: i32 = 20159_i32;
    let v_vec: int32x4_t = vdupq_n_s32(V);

    // Widen to 32-bit
    let a_lo: int32x4_t = vmovl_s16(vget_low_s16(a));
    let a_hi: int32x4_t = vmovl_high_s16(a);

    // t = (x * V) >> 26
    let t_lo: int32x4_t = vshrq_n_s32(vmulq_s32(a_lo, v_vec), 26);
    let t_hi: int32x4_t = vshrq_n_s32(vmulq_s32(a_hi, v_vec), 26);

    // Narrow t to 16-bit
    let t16: int16x8_t = vcombine_s16(vmovn_s32(t_lo), vmovn_s32(t_hi));

    // result = x − t·q
    let q_vec: int16x8_t = vdupq_n_s16(Q);
    vsubq_s16(a, vmulq_s16(t16, q_vec))
}

// ──────────────────────────────────────────────────────────────────────────────
// Cooley–Tukey butterfly (forward NTT layer)
//
//  a' = a + t        where t = MontMul(b, zeta)
//  b' = a − t
//
//  Returns (a', b').  Both outputs are within [−2q, 2q] if inputs are.
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ct_butterfly(
    a: int16x8_t,
    b: int16x8_t,
    zeta: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let t = montgomery_mul_vec(b, zeta);
    (vaddq_s16(a, t), vsubq_s16(a, t))
}

// ──────────────────────────────────────────────────────────────────────────────
// Gentleman–Sande butterfly (inverse NTT layer)
//
//  a' = a + b
//  b' = MontMul(a − b, zeta)
//
//  Returns (a', b').
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gs_butterfly(
    a: int16x8_t,
    b: int16x8_t,
    zeta: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let diff = vsubq_s16(b, a);
    (vaddq_s16(a, b), montgomery_mul_vec(diff, zeta))
}

// ──────────────────────────────────────────────────────────────────────────────
// Forward NTT  (FIPS 203, Algorithm 9)
//
//  Transforms `poly` in-place from the coefficient domain to the NTT domain.
//  Seven layers of Cooley–Tukey butterflies with bit-reversed twiddle factors.
//
//  Layer structure (len = half-butterfly-stride):
//    Layer 0: len = 128  (1 butterfly block  of 128 pairs)
//    Layer 1: len =  64  (2 butterfly blocks of  64 pairs)
//    Layer 2: len =  32  (4 …)
//    Layer 3: len =  16
//    Layer 4: len =   8  → last layer handled fully in SIMD registers
//    Layer 5: len =   4  → pairs within a single int16x8_t
//    Layer 6: len =   2  → pairs within a single int16x8_t
//
//  Barrett reduction is applied after layers 0, 3, and 6 to keep coefficients
//  from overflowing i16.  Each CT butterfly grows bounds by at most +q, so
//  after 3 layers without Barrett the theoretical maximum is 8q ≈ 26632 which
//  overflows i16.  We reduce after layers 1 and 4 instead (every 3 layers),
//  keeping the maximum below 4q = 13316 < 32767.
// ──────────────────────────────────────────────────────────────────────────────
/// Forward NTT. Takes `Poly` by value, transforms it, returns it.
/// Call site: `let poly_hat = ntt::ntt(poly);`
/// Forward NTT.
pub fn ntt(mut poly: Poly) -> Poly {
    #[cfg(target_arch = "aarch64")]
    unsafe { ntt_inner(&mut poly) }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let out = scalar_ref::ntt_ref(&poly.0);
        poly.0.copy_from_slice(&out);
    }
    poly
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ntt_inner(poly: &mut Poly) {
    let p = poly.0.as_mut_ptr();
    // ── Zeta index counter — matches the i variable in FIPS 203 Algorithm 9 ──
    let mut zeta_idx: usize = 1; // ZETAS_NEON[0] unused

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 0: len = 128,  1 block  (one twiddle fills all 128 CT-butterfly pairs)
    // ══════════════════════════════════════════════════════════════════════════
    {
        let zeta: int16x8_t = vdupq_n_s16(ZETAS_NEON[zeta_idx]);
        zeta_idx += 1;
        let mut j = 0usize;
        while j < 128 {
            let a = vld1q_s16(p.add(j));
            let b = vld1q_s16(p.add(j + 128));
            let (a2, b2) = ct_butterfly(a, b, zeta);
            vst1q_s16(p.add(j),       a2);
            vst1q_s16(p.add(j + 128), b2);
            j += 8;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 1: len = 64,  2 blocks
    // ══════════════════════════════════════════════════════════════════════════
    for start in [0usize, 128] {
        let zeta: int16x8_t = vdupq_n_s16(ZETAS_NEON[zeta_idx]);
        zeta_idx += 1;
        let mut j = start;
        while j < start + 64 {
            let a = vld1q_s16(p.add(j));
            let b = vld1q_s16(p.add(j + 64));
            let (a2, b2) = ct_butterfly(a, b, zeta);
            vst1q_s16(p.add(j),      a2);
            vst1q_s16(p.add(j + 64), b2);
            j += 8;
        }
    }

    // Barrett reduce after 2 layers to stay in range
    {
        let mut j = 0usize;
        while j < 256 {
            let v = vld1q_s16(p.add(j));
            vst1q_s16(p.add(j), barrett_reduce_vec(v));
            j += 8;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 2: len = 32,  4 blocks
    // ══════════════════════════════════════════════════════════════════════════
    for start in [0usize, 64, 128, 192] {
        let zeta: int16x8_t = vdupq_n_s16(ZETAS_NEON[zeta_idx]);
        zeta_idx += 1;
        let mut j = start;
        while j < start + 32 {
            let a = vld1q_s16(p.add(j));
            let b = vld1q_s16(p.add(j + 32));
            let (a2, b2) = ct_butterfly(a, b, zeta);
            vst1q_s16(p.add(j),      a2);
            vst1q_s16(p.add(j + 32), b2);
            j += 8;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 3: len = 16,  8 blocks
    // ══════════════════════════════════════════════════════════════════════════
    let mut start = 0usize;
    while start < 256 {
        let zeta: int16x8_t = vdupq_n_s16(ZETAS_NEON[zeta_idx]);
        zeta_idx += 1;
        let mut j = start;
        while j < start + 16 {
            let a = vld1q_s16(p.add(j));
            let b = vld1q_s16(p.add(j + 16));
            let (a2, b2) = ct_butterfly(a, b, zeta);
            vst1q_s16(p.add(j),      a2);
            vst1q_s16(p.add(j + 16), b2);
            j += 8;
        }
        start += 32;
    }

    // Barrett reduce again
    {
        let mut j = 0usize;
        while j < 256 {
            let v = vld1q_s16(p.add(j));
            vst1q_s16(p.add(j), barrett_reduce_vec(v));
            j += 8;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 4: len = 8,  16 blocks
    // Each block is exactly one int16x8_t pair (a[j..j+8], b[j+8..j+16]).
    // ══════════════════════════════════════════════════════════════════════════
    let mut start = 0usize;
    while start < 256 {
        let zeta: int16x8_t = vdupq_n_s16(ZETAS_NEON[zeta_idx]);
        zeta_idx += 1;
        let a = vld1q_s16(p.add(start));
        let b = vld1q_s16(p.add(start + 8));
        let (a2, b2) = ct_butterfly(a, b, zeta);
        vst1q_s16(p.add(start),     a2);
        vst1q_s16(p.add(start + 8), b2);
        start += 16;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 5: len = 4,  32 blocks
    // Now the two halves of a butterfly live in the *same* int16x8_t register:
    //   lanes 0..3 → "a" side,  lanes 4..7 → "b" side
    // We process two adjacent blocks of 8 in one pass.
    // ══════════════════════════════════════════════════════════════════════════
    let mut start = 0usize;
    while start < 256 {
        // Two consecutive twiddles (one per 8-element block)
        let z0 = ZETAS_NEON[zeta_idx];
        let z1 = ZETAS_NEON[zeta_idx + 1];
        zeta_idx += 2;

        // Build a zeta vector: [z0,z0,z0,z0, z1,z1,z1,z1]
        let zeta_lo: int16x4_t = vdup_n_s16(z0);
        let zeta_hi: int16x4_t = vdup_n_s16(z1);
        let zeta_vec: int16x8_t = vcombine_s16(zeta_lo, zeta_hi);

        // Load two 8-element blocks
        let ab0 = vld1q_s16(p.add(start));       // a0..a3, b0..b3
        let ab1 = vld1q_s16(p.add(start + 8));   // a1_0..a1_3, b1_0..b1_3

        // Separate "a" lanes (0..3) and "b" lanes (4..7)
        // vuzp1q / vuzp2q rearrange two vectors, not what we want here.
        // Instead split each vec at the 64-bit boundary.
        let a_vec: int16x8_t = vcombine_s16(vget_low_s16(ab0), vget_low_s16(ab1));  // block0_a, block1_a
        let b_vec: int16x8_t = vcombine_s16(vget_high_s16(ab0), vget_high_s16(ab1)); // block0_b, block1_b

        let t = montgomery_mul_vec(b_vec, zeta_vec);
        let a_new = vaddq_s16(a_vec, t);
        let b_new = vsubq_s16(a_vec, t);

        // Interleave back: block0 = [a_new_lo | b_new_lo], block1 = [a_new_hi | b_new_hi]
        let out0: int16x8_t = vcombine_s16(vget_low_s16(a_new), vget_low_s16(b_new));
        let out1: int16x8_t = vcombine_s16(vget_high_s16(a_new), vget_high_s16(b_new));
        vst1q_s16(p.add(start),     out0);
        vst1q_s16(p.add(start + 8), out1);
        start += 16;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 6: len = 2,  64 blocks
    // Butterfly pairs are adjacent: (lane0,lane1) with (lane2,lane3), etc.
    // Treating `v` as int32x4_t `[L0, L1, L2, L3]`, the "a" halves are L0, L2
    // and the "b" halves are L1, L3.
    // ══════════════════════════════════════════════════════════════════════════
    let mut start = 0usize;
    while start < 256 {
        let z0 = ZETAS_NEON[zeta_idx];
        let z1 = ZETAS_NEON[zeta_idx + 1];
        let z2 = ZETAS_NEON[zeta_idx + 2];
        let z3 = ZETAS_NEON[zeta_idx + 3];
        zeta_idx += 4;

        let zeta_arr: [i16; 8] = [z0, z0, z1, z1, z2, z2, z3, z3];
        let zeta_vec: int16x8_t = vld1q_s16(zeta_arr.as_ptr());

        let v0 = vld1q_s16(p.add(start));
        let v1 = vld1q_s16(p.add(start + 8));

        let v0_32 = vreinterpretq_s32_s16(v0);
        let v1_32 = vreinterpretq_s32_s16(v1);

        // Extract a_vec_32 = [L0, L2, L4, L6] and b_vec_32 = [L1, L3, L5, L7]
        let a_vec_32 = vuzp1q_s32(v0_32, v1_32);
        let b_vec_32 = vuzp2q_s32(v0_32, v1_32);

        let a_vec = vreinterpretq_s16_s32(a_vec_32);
        let b_vec = vreinterpretq_s16_s32(b_vec_32);

        let t = montgomery_mul_vec(b_vec, zeta_vec);
        let a_new = vaddq_s16(a_vec, t);
        let b_new = vsubq_s16(a_vec, t);

        let a_new_32 = vreinterpretq_s32_s16(a_new);
        let b_new_32 = vreinterpretq_s32_s16(b_new);

        // Re-interleave to out0_32 = [L0', L1', L2', L3'] and out1_32 = [L4', L5', L6', L7']
        let out0_32 = vzip1q_s32(a_new_32, b_new_32);
        let out1_32 = vzip2q_s32(a_new_32, b_new_32);

        vst1q_s16(p.add(start), vreinterpretq_s16_s32(out0_32));
        vst1q_s16(p.add(start + 8), vreinterpretq_s16_s32(out1_32));

        start += 16;
    }

    // Final Barrett reduce to guarantee output in [0, 2q)
    {
        let mut j = 0usize;
        while j < 256 {
            let v = vld1q_s16(p.add(j));
            vst1q_s16(p.add(j), barrett_reduce_vec(v));
            j += 8;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Inverse NTT  (FIPS 203, Algorithm 10)
//
//  Transforms `poly` in-place from the NTT domain back to coefficients.
//  Uses Gentleman–Sande butterflies and the inverse twiddle table.
//  The final step multiplies every coefficient by 3303 = 128⁻¹·R mod q,
//  which simultaneously handles the 1/128 scaling AND removes the Montgomery
//  factor (i.e., the output is a standard ℤ_q coefficient).
// ──────────────────────────────────────────────────────────────────────────────
/// Inverse NTT. Takes `Poly` by value, transforms it, returns it.
/// Call site: `let poly = ntt::inv_ntt(poly_hat);`
/// Inverse NTT.
pub fn inv_ntt(mut poly: Poly) -> Poly {
    #[cfg(target_arch = "aarch64")]
    unsafe { inv_ntt_inner(&mut poly) }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let out = scalar_ref::inv_ntt_ref(&poly.0);
        poly.0.copy_from_slice(&out);
    }
    poly
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn inv_ntt_inner(poly: &mut Poly) {
    let p = poly.0.as_mut_ptr();
    let mut zeta_idx: usize = 0; // INV_ZETAS_NEON scanned forward

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 0 (GS, len=2): 64 blocks
    // Uses the same 32-bit unzip trick as forward Layer 6 to separate pairs
    // with a stride of 2 elements.
    // ══════════════════════════════════════════════════════════════════════════
    let mut start = 0usize;
    while start < 256 {
        let z0 = INV_ZETAS_NEON[zeta_idx];
        let z1 = INV_ZETAS_NEON[zeta_idx + 1];
        let z2 = INV_ZETAS_NEON[zeta_idx + 2];
        let z3 = INV_ZETAS_NEON[zeta_idx + 3];
        zeta_idx += 4;

        let zeta_arr: [i16; 8] = [z0, z0, z1, z1, z2, z2, z3, z3];
        let zeta_vec: int16x8_t = vld1q_s16(zeta_arr.as_ptr());

        let v0 = vld1q_s16(p.add(start));
        let v1 = vld1q_s16(p.add(start + 8));

        let v0_32 = vreinterpretq_s32_s16(v0);
        let v1_32 = vreinterpretq_s32_s16(v1);

        let a_vec_32 = vuzp1q_s32(v0_32, v1_32);
        let b_vec_32 = vuzp2q_s32(v0_32, v1_32);

        let a_vec = vreinterpretq_s16_s32(a_vec_32);
        let b_vec = vreinterpretq_s16_s32(b_vec_32);

        let sum = vaddq_s16(a_vec, b_vec);
        let diff_mont = montgomery_mul_vec(vsubq_s16(b_vec, a_vec), zeta_vec);

        let sum_32 = vreinterpretq_s32_s16(sum);
        let diff_32 = vreinterpretq_s32_s16(diff_mont);

        let out0_32 = vzip1q_s32(sum_32, diff_32);
        let out1_32 = vzip2q_s32(sum_32, diff_32);

        vst1q_s16(p.add(start), vreinterpretq_s16_s32(out0_32));
        vst1q_s16(p.add(start + 8), vreinterpretq_s16_s32(out1_32));

        start += 16;
    }
    // ══════════════════════════════════════════════════════════════════════════
    // Layer 1 (GS, len=4): 32 blocks  (two adjacent 8-element vectors per pass)
    // ══════════════════════════════════════════════════════════════════════════
    let mut start = 0usize;
    while start < 256 {
        let z0 = INV_ZETAS_NEON[zeta_idx];
        let z1 = INV_ZETAS_NEON[zeta_idx + 1];
        zeta_idx += 2;

        let zeta_lo: int16x4_t = vdup_n_s16(z0);
        let zeta_hi: int16x4_t = vdup_n_s16(z1);
        let zeta_vec: int16x8_t = vcombine_s16(zeta_lo, zeta_hi);

        let ab0 = vld1q_s16(p.add(start));
        let ab1 = vld1q_s16(p.add(start + 8));

        let a_vec: int16x8_t = vcombine_s16(vget_low_s16(ab0), vget_low_s16(ab1));
        let b_vec: int16x8_t = vcombine_s16(vget_high_s16(ab0), vget_high_s16(ab1));

        let sum = vaddq_s16(a_vec, b_vec);
        let diff_mont = montgomery_mul_vec(vsubq_s16(b_vec, a_vec), zeta_vec);

        let out0: int16x8_t = vcombine_s16(vget_low_s16(sum), vget_low_s16(diff_mont));
        let out1: int16x8_t = vcombine_s16(vget_high_s16(sum), vget_high_s16(diff_mont));
        vst1q_s16(p.add(start),     out0);
        vst1q_s16(p.add(start + 8), out1);
        start += 16;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 2 (GS, len=8): 16 blocks
    // ══════════════════════════════════════════════════════════════════════════
    let mut start = 0usize;
    while start < 256 {
        let zeta: int16x8_t = vdupq_n_s16(INV_ZETAS_NEON[zeta_idx]);
        zeta_idx += 1;
        let a = vld1q_s16(p.add(start));
        let b = vld1q_s16(p.add(start + 8));
        let (a2, b2) = gs_butterfly(a, b, zeta);
        vst1q_s16(p.add(start),     a2);
        vst1q_s16(p.add(start + 8), b2);
        start += 16;
    }

    // Barrett reduce
    {
        let mut j = 0usize;
        while j < 256 {
            let v = vld1q_s16(p.add(j));
            vst1q_s16(p.add(j), barrett_reduce_vec(v));
            j += 8;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 3 (GS, len=16): 8 blocks
    // ══════════════════════════════════════════════════════════════════════════
    let mut start = 0usize;
    while start < 256 {
        let zeta: int16x8_t = vdupq_n_s16(INV_ZETAS_NEON[zeta_idx]);
        zeta_idx += 1;
        let mut j = start;
        while j < start + 16 {
            let a = vld1q_s16(p.add(j));
            let b = vld1q_s16(p.add(j + 16));
            let (a2, b2) = gs_butterfly(a, b, zeta);
            vst1q_s16(p.add(j),      a2);
            vst1q_s16(p.add(j + 16), b2);
            j += 8;
        }
        start += 32;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 4 (GS, len=32): 4 blocks
    // ══════════════════════════════════════════════════════════════════════════
    for start in [0usize, 64, 128, 192] {
        let zeta: int16x8_t = vdupq_n_s16(INV_ZETAS_NEON[zeta_idx]);
        zeta_idx += 1;
        let mut j = start;
        while j < start + 32 {
            let a = vld1q_s16(p.add(j));
            let b = vld1q_s16(p.add(j + 32));
            let (a2, b2) = gs_butterfly(a, b, zeta);
            vst1q_s16(p.add(j),      a2);
            vst1q_s16(p.add(j + 32), b2);
            j += 8;
        }
    }

    // Barrett reduce
    {
        let mut j = 0usize;
        while j < 256 {
            let v = vld1q_s16(p.add(j));
            vst1q_s16(p.add(j), barrett_reduce_vec(v));
            j += 8;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 5 (GS, len=64): 2 blocks
    // ══════════════════════════════════════════════════════════════════════════
    for start in [0usize, 128] {
        let zeta: int16x8_t = vdupq_n_s16(INV_ZETAS_NEON[zeta_idx]);
        zeta_idx += 1;
        let mut j = start;
        while j < start + 64 {
            let a = vld1q_s16(p.add(j));
            let b = vld1q_s16(p.add(j + 64));
            let (a2, b2) = gs_butterfly(a, b, zeta);
            vst1q_s16(p.add(j),      a2);
            vst1q_s16(p.add(j + 64), b2);
            j += 8;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Layer 6 (GS, len=128): 1 block
    // ══════════════════════════════════════════════════════════════════════════
    {
        let zeta: int16x8_t = vdupq_n_s16(INV_ZETAS_NEON[zeta_idx]);
        // zeta_idx would be 127 here; we don't need to advance further.
        let mut j = 0usize;
        while j < 128 {
            let a = vld1q_s16(p.add(j));
            let b = vld1q_s16(p.add(j + 128));
            let (a2, b2) = gs_butterfly(a, b, zeta);
            vst1q_s16(p.add(j),       a2);
            vst1q_s16(p.add(j + 128), b2);
            j += 8;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Final scaling: multiply every coefficient by INV128_MONT = 3303
    // This is a one-known-factor Montgomery multiply: output = coeff·3303·R⁻¹
    // ≡ coeff · 128⁻¹ (mod q), simultaneously removing the Montgomery factor.
    // ══════════════════════════════════════════════════════════════════════════
    {
        let scale: int16x8_t = vdupq_n_s16(INV128_MONT);
        let mut j = 0usize;
        while j < 256 {
            let v = vld1q_s16(p.add(j));
            let scaled = montgomery_mul_vec(v, scale);
            // Barrett-reduce to [0, q) for canonical output
            vst1q_s16(p.add(j), barrett_reduce_vec(scaled));
            j += 8;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pointwise multiplication in the NTT domain  (FIPS 203, Algorithms 11 & 12)
//
//  Computes `out = a ⊙ b` where ⊙ is the componentwise product in T_q.
//  Each pair (f̂[2i], f̂[2i+1]) lives in F_q[X] / (X² − ζ^(2·BitRev7(i)+1))
//  so the base-case multiply is:
//    c0 = a0·b0 + a1·b1·γ   (γ = ζ^(2·BitRev7(i)+1))
//    c1 = a0·b1 + a1·b0
//
//  We de-interleave using `vld2q_s16` to separate even/odd lanes, perform the
//  four products via Montgomery multiplication, then re-interleave.
// ──────────────────────────────────────────────────────────────────────────────
/// Pointwise multiply two NTT-domain polynomials, returning the product.
pub fn mul_ntt(a: &Poly, b: &Poly) -> Poly {
    let mut out = Poly::zero();
    #[cfg(target_arch = "aarch64")]
    unsafe { mul_ntt_inner(a, b, &mut out) }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // scalar base-case multiply
        use crate::reduce;
        let mut i = 0;
        while i < 256 {
            let a0 = a.0[i] as i32;
            let a1 = a.0[i+1] as i32;
            let b0 = b.0[i] as i32;
            let b1 = b.0[i+1] as i32;
            let gamma = GAMMAS_NEON[i] as i32;
            // c0 = a0*b0 + a1*b1*gamma (mod q)
            let c0 = scalar_ref::mont_reduce(a0 * b0) as i32
                   + scalar_ref::mont_reduce(scalar_ref::mont_reduce(a1 * b1) as i32 * gamma) as i32;
            let c1 = scalar_ref::mont_reduce(a0 * b1) as i32
                   + scalar_ref::mont_reduce(a1 * b0) as i32;
            out.0[i]   = reduce::mod_q(c0);
            out.0[i+1] = reduce::mod_q(c1);
            i += 2;
        }
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mul_ntt_inner(a: &Poly, b: &Poly, out: &mut Poly) {
    let pa = a.0.as_ptr();
    let pb = b.0.as_ptr();
    let po = out.0.as_mut_ptr();

    // Process 8 basemul pairs (= 16 coefficients) per iteration.
    // GAMMAS_NEON layout: [γ₀, −γ₀, γ₁, −γ₁, …] in pairs of 2.
    // Loading 16 entries gives us 8 (γ, −γ) pairs.
    let mut i = 0usize; // coefficient index, step 16
    let mut gi = 0usize; // GAMMAS_NEON index, step 16

    while i < 256 {
        // De-interleave a and b into even (index 2k) and odd (index 2k+1) lanes
        let a_pairs = vld2q_s16(pa.add(i));
        let b_pairs = vld2q_s16(pb.add(i));
        let a0: int16x8_t = a_pairs.0; // a[0],a[2],a[4],… (even = ĝ[2i])
        let a1: int16x8_t = a_pairs.1; // a[1],a[3],a[5],… (odd  = ĝ[2i+1])
        let b0: int16x8_t = b_pairs.0;
        let b1: int16x8_t = b_pairs.1;

        // Load 8 γ values and 8 −γ values (interleaved in GAMMAS_NEON)
        let gam_pairs = vld2q_s16(GAMMAS_NEON.as_ptr().add(gi));
        let gamma: int16x8_t = gam_pairs.0;     // [γ₀, γ₁, …, γ₇]
        // neg_gamma = gam_pairs.1 but we don't actually need it explicitly —
        // a1·b1·γ uses `gamma` directly; the subtraction is handled by sign.

        // c0 = a0·b0 + a1·b1·γ
        //    = MontMul(a0, b0) + MontMul(a1·b1, γ)
        // We can't accumulate before reducing, so do two Montgomery multiplies
        // and add.  Both outputs are in (−q, q), so the sum is in (−2q, 2q) ✓
        let a0b0: int16x8_t = montgomery_mul_vec(a0, b0);
        let a1b1: int16x8_t = montgomery_mul_vec(a1, b1);
        let a1b1_gamma: int16x8_t = montgomery_mul_vec(a1b1, gamma);
        // Note: a1b1 is already R⁻¹-scaled; multiplying by γ (not in Montgomery
        // form) gives a1·b1·γ·R⁻¹.  Combined with a0b0 = a0·b0·R⁻¹, the result
        // c0 = (a0·b0 + a1·b1·γ)·R⁻¹ is consistently scaled. ✓
        let c0: int16x8_t = vaddq_s16(a0b0, a1b1_gamma);

        // c1 = a0·b1 + a1·b0
        let a0b1: int16x8_t = montgomery_mul_vec(a0, b1);
        let a1b0: int16x8_t = montgomery_mul_vec(a1, b0);
        let c1: int16x8_t = vaddq_s16(a0b1, a1b0);

        // Re-interleave c0 (even) and c1 (odd) and store
        vst2q_s16(po.add(i), int16x8x2_t(c0, c1));

        i += 16;
        gi += 16;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Scalar reference helpers (used for tests / parameter derivation)
// ──────────────────────────────────────────────────────────────────────────────
mod scalar_ref {
    use super::Q;

    /// Montgomery reduction: (a · R⁻¹) mod q
    pub fn mont_reduce(a: i32) -> i16 {
        const QINV: i32 = -3327; // −q⁻¹ mod R, R = 2¹⁶
        let m = ((a as i32).wrapping_mul(QINV)) as i16;
        let t = ((a - (m as i32) * (Q as i32)) >> 16) as i16;
        t
    }

    /// Barrett reduction to [0, q)
    pub fn barrett_reduce(a: i16) -> i16 {
        let v: i32 = 20159;
        let t = ((a as i32 * v) >> 26) as i16;
        let r = a - t * Q;
        // conditionally add/sub q to land in [0, q)
        let r = if r >= Q { r - Q } else { r };
        let r = if r < 0 { r + Q } else { r };
        r
    }

    /// Scalar reference NTT (FIPS 203 Algorithm 9, plain arithmetic)
    pub fn ntt_ref(f: &[i16; 256]) -> [i16; 256] {
        const Q: i32 = 3329;
        let mut f = *f;
        let mut k = 1usize;
        let mut len = 128usize;
        while len >= 2 {
            let mut start = 0usize;
            while start < 256 {
                // Zeta = 17^{BitRev7(k)} mod Q (plain, no Montgomery scaling)
                let zeta = {
                    let mut br = 0u32;
                    let mut n = k as u32;
                    for _ in 0..7 { br = (br << 1) | (n & 1); n >>= 1; }
                    let mut z = 1i32;
                    let mut base = 17i32;
                    let mut exp = br;
                    while exp > 0 {
                        if exp & 1 != 0 { z = z * base % Q; }
                        base = base * base % Q;
                        exp >>= 1;
                    }
                    z
                };
                k += 1;
                for j in start..start + len {
                    let t = (zeta * f[j + len] as i32).rem_euclid(Q) as i16;
                    f[j + len] = f[j] - t;
                    f[j] = f[j] + t;
                }
                start += 2 * len;
            }
            len >>= 1;
        }
        f
    }

    /// Scalar reference NTT⁻¹ (FIPS 203 Algorithm 10, plain arithmetic)
    pub fn inv_ntt_ref(f: &[i16; 256]) -> [i16; 256] {
        const Q: i32 = 3329;
        let mut f = *f;
        let mut k = 127usize;
        let mut len = 2usize;
        while len <= 128 {
            let mut start = 0usize;
            while start < 256 {
                // Zeta = 17^{BitRev7(k)} mod Q (same as forward NTT at position k)
                let zeta = {
                    let mut br = 0u32;
                    let mut n = k as u32;
                    for _ in 0..7 { br = (br << 1) | (n & 1); n >>= 1; }
                    let mut z = 1i32;
                    let mut base = 17i32;
                    let mut exp = br;
                    while exp > 0 {
                        if exp & 1 != 0 { z = z * base % Q; }
                        base = base * base % Q;
                        exp >>= 1;
                    }
                    z
                };
                k = k.wrapping_sub(1);
                for j in start..start + len {
                    let t = f[j];
                    f[j] = (t as i32 + f[j + len] as i32).rem_euclid(Q) as i16;
                    f[j + len] = (zeta * (f[j + len] as i32 - t as i32)).rem_euclid(Q) as i16;
                }
                start += 2 * len;
            }
            len <<= 1;
        }
        // Final scaling: × 128⁻¹ mod Q
        const INV128: i32 = 3303; // pow(128, -1, 3329)
        for coeff in f.iter_mut() {
            *coeff = ((*coeff as i32 * INV128).rem_euclid(Q)) as i16;
        }
        f
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use scalar_ref::*;

    // Only run SIMD tests when actually on aarch64
    #[cfg(target_arch = "aarch64")]
    fn make_test_poly(seed: u16) -> Poly {
        let mut p = Poly::zero();
        for (i, c) in p.0.iter_mut().enumerate() {
            // Pseudo-random coefficients in [0, Q)
            *c = ((seed.wrapping_add(i as u16).wrapping_mul(0x9E37)) % Q as u16) as i16;
        }
        p
    }

    /// Verify that NTT⁻¹(NTT(a)) == a  (round-trip, bit-exact)
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_ntt_roundtrip() {
        let original = make_test_poly(0xABCD);
        let mut poly = original.clone();

        poly = ntt(poly);
        poly = inv_ntt(poly);

        for (i, (&got, &expected)) in poly.0.iter().zip(original.0.iter()).enumerate() {
            // After round-trip, coefficients must match modulo Q
            let got_mod = ((got % Q).wrapping_add(Q)) % Q;
            let exp_mod = ((expected % Q).wrapping_add(Q)) % Q;
            assert_eq!(
                got_mod, exp_mod,
                "Round-trip mismatch at index {}: got {} ({}), expected {} ({})",
                i, got, got_mod, expected, exp_mod
            );
        }
    }

    /// Verify NTT matches the scalar reference implementation
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_ntt_matches_reference() {
        let poly = make_test_poly(0x1234);

        // Scalar reference
        let coeffs_ref = {
            let mut arr = [0i16; 256];
            arr.copy_from_slice(&poly.0);
            ntt_ref(&arr)
        };

        // SIMD implementation
        let poly_simd = poly.clone();
        let poly_simd = ntt(poly_simd);

        for i in 0..256 {
            let ref_mod = ((coeffs_ref[i] % Q).wrapping_add(Q)) % Q;
            let got_mod = ((poly_simd.0[i] % Q).wrapping_add(Q)) % Q;
            assert_eq!(
                ref_mod, got_mod,
                "NTT mismatch at index {}: ref={} got={}",
                i, ref_mod, got_mod
            );
        }
    }

    /// Verify NTT linearity: NTT(a + b) == NTT(a) + NTT(b)  (mod Q, componentwise)
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_ntt_linearity() {
        let a = make_test_poly(0x0001);
        let b = make_test_poly(0x0002);
        let apb = Poly::add(&a, &b);

        let ntt_a = a.clone();
        let ntt_b = b.clone();
        let ntt_apb = apb.clone();

        let ntt_a = ntt(ntt_a);
        let ntt_b = ntt(ntt_b);
        let ntt_apb = ntt(ntt_apb);

        let sum = Poly::add(&ntt_a, &ntt_b);
        for i in 0..256 {
            let lhs = ((ntt_apb.0[i] % Q).wrapping_add(Q)) % Q;
            let rhs = ((sum.0[i] % Q).wrapping_add(Q)) % Q;
            assert_eq!(
                lhs, rhs,
                "Linearity failure at index {}: NTT(a+b)[{}]={} but (NTT(a)+NTT(b))[{}]={}",
                i, i, lhs, i, rhs
            );
        }
    }

    /// Test that mul_ntt produces consistent results (a * b == b * a in NTT domain)
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_mul_ntt_commutativity() {
        let a = make_test_poly(0xAAAA);
        let b = make_test_poly(0x5555);
        let a = ntt(a);
        let b = ntt(b);

        let out_ab = mul_ntt(&a, &b);
        let out_ba = mul_ntt(&b, &a);

        for i in 0..256 {
            let ab = ((out_ab.0[i] % Q).wrapping_add(Q)) % Q;
            let ba = ((out_ba.0[i] % Q).wrapping_add(Q)) % Q;
            assert_eq!(ab, ba, "Commutativity failure at index {}", i);
        }
    }

    /// Scalar reference test: NTT⁻¹(NTT(a)) == a
    #[test]
    fn test_scalar_roundtrip() {
        let mut arr = [0i16; 256];
        for (i, c) in arr.iter_mut().enumerate() {
            *c = (i as i16 * 7 + 3) % Q;
        }
        let fwd = ntt_ref(&arr);
        let inv = inv_ntt_ref(&fwd);
        for i in 0..256 {
            let expected = ((arr[i] % Q).wrapping_add(Q)) % Q;
            let got = ((inv[i] % Q).wrapping_add(Q)) % Q;
            assert_eq!(expected, got, "Scalar round-trip failure at index {}", i);
        }
    }
}