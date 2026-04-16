use crate::params::MlKem768;
use crate::poly::Poly;
use crate::hash::prf_shake256;
use crate::ntt;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake128;

/// SampleNTT: rejection sample coefficients mod q from XOF stream
pub fn sample_ntt(seed: &[u8; 32], i: u8, j: u8) -> Poly {
    let mut coeffs = [0i16; MlKem768::N];
    let mut hasher = Shake128::default();
    hasher.update(seed);
    hasher.update(&[i, j]);
    let mut reader = hasher.finalize_xof();

    // SHAKE128 rate is 168 bytes; keep a small carry-over buffer for leftover bytes.
    let mut buf = [0u8; 170]; // 168 + up to 2 leftover bytes
    let mut leftover = 0usize;
    let mut ctr = 0usize;

    while ctr < MlKem768::N {
        reader.read(&mut buf[leftover..leftover + 168]);
        let total = leftover + 168;
        let mut idx = 0usize;

        while idx + 3 <= total && ctr < MlKem768::N {
            let a0 = (buf[idx] as u16) | (((buf[idx + 1] as u16) & 0x0F) << 8);
            let a1 = ((buf[idx + 1] as u16) >> 4) | ((buf[idx + 2] as u16) << 4);
            idx += 3;

            if a0 < MlKem768::Q as u16 {
                coeffs[ctr] = a0 as i16;
                ctr += 1;
                if ctr == MlKem768::N {
                    break;
                }
            }
            if a1 < MlKem768::Q as u16 {
                coeffs[ctr] = a1 as i16;
                ctr += 1;
            }
        }

        // Preserve up to two leftover bytes for the next block.
        leftover = total - idx;
        if leftover > 0 {
            buf.copy_within(idx..total, 0);
        }
    }

    Poly(coeffs)
}

/// SamplePolyCBD(eta): centered binomial distribution from PRF stream
pub fn sample_poly_cbd_eta(seed: &[u8; 32], nonce: u8, eta: usize) -> Poly {
    debug_assert_eq!(eta, MlKem768::ETA2, "only eta=2 supported");

    // Generate PRF output: eta * N / 4 bytes = 128 bytes for eta=2, N=256.
    let mut buf = [0u8; MlKem768::ETA2 * MlKem768::N / 4];
    prf_shake256(seed, nonce, &mut buf);

    let mut coeffs = [0i16; MlKem768::N];
    let mut off = 0usize;

    // Process 4 bytes -> 8 coefficients (CBD for eta=2).
    for chunk in 0..(MlKem768::N / 8) {
        let t = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        off += 4;

        // d holds pairwise bit sums: counts of set bits in each 2-bit group.
        let d = (t & 0x5555_5555).wrapping_add((t >> 1) & 0x5555_5555);

        let base = chunk * 8;
        for j in 0..8 {
            let a = ((d >> (4 * j)) & 0x3) as i16;
            let b = ((d >> (4 * j + 2)) & 0x3) as i16;
            coeffs[base + j] = a - b;
        }
    }

    Poly(coeffs)
}
