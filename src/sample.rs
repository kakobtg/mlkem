use crate::hash::prf_shake256;
use crate::params::MlKem768;
use crate::poly::Poly;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake128;

/// SampleNTT: rejection sample coefficients mod q from XOF stream
pub fn sample_ntt(seed: &[u8; 32], i: u8, j: u8) -> Poly {
    let mut coeffs = [0i16; MlKem768::N];
    let mut hasher = Shake128::default();
    hasher.update(seed);
    hasher.update(&[j, i]); // FIPS 203 requires (rho || j || i), not (rho || i || j)
    let mut reader = hasher.finalize_xof();

    let mut buf = [0u8; 170]; // SHAKE128 rate (168) + up to 2 leftover bytes
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

        leftover = total - idx;
        if leftover > 0 {
            buf.copy_within(idx..total, 0);
        }
    }

    Poly(coeffs)
}

/// SamplePolyCBD(eta): centered binomial distribution from PRF stream.
/// The bit trick below only works for eta=2 (ML-KEM-768's only value), so
/// this asserts unconditionally rather than via `debug_assert!` — a wrong
/// eta must fail loudly, not silently sample the wrong distribution.
pub fn sample_poly_cbd_eta(seed: &[u8; 32], nonce: u8, eta: usize) -> Poly {
    assert_eq!(eta, MlKem768::ETA2, "sample_poly_cbd_eta only implements eta=2");

    let mut buf = [0u8; MlKem768::ETA2 * MlKem768::N / 4];
    prf_shake256(seed, nonce, &mut buf);

    let mut coeffs = [0i16; MlKem768::N];
    let mut off = 0usize;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_ntt_bounds() {
        let seed = [0xAA; 32];
        let poly = sample_ntt(&seed, 0, 0);
        for &coef in poly.0.iter() {
            assert!(
                coef >= 0 && coef < MlKem768::Q as i16,
                "NTT sample out of bounds: {}",
                coef
            );
        }
    }

    #[test]
    fn test_sample_poly_cbd_bounds() {
        let seed = [0xBB; 32];
        let poly = sample_poly_cbd_eta(&seed, 0, MlKem768::ETA2);
        for &coef in poly.0.iter() {
            assert!(
                coef >= -2 && coef <= 2,
                "CBD sample out of bounds: {}",
                coef
            );
        }
    }
}
