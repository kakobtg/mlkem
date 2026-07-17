use crate::params::MlKem768;
use crate::poly::Poly;

/// ByteEncode_d / ByteDecode_d for d in {1..12} (you will need specific ds)
pub fn byte_encode<const D: usize>(p: &Poly, out: &mut [u8]) {
    debug_assert!(D > 0 && D <= 12);

    let mut acc: u64 = 0;
    let mut bits: usize = 0;
    let mut idx: usize = 0;
    let mask: u64 = (1u64 << D) - 1;

    for &coef in p.0.iter() {
        let mut c = coef;
        
        // For D=12, Kyber requires coefficients modulo Q strictly in [0, Q)
        if D == 12 {
            c = c % (MlKem768::Q as i16);
            if c < 0 {
                c += MlKem768::Q as i16;
            }
        }
        
        acc |= ((c as u16 as u64) & mask) << bits;
        bits += D;

        while bits >= 8 {
            out[idx] = acc as u8;
            acc >>= 8;
            bits -= 8;
            idx += 1;
        }
    }

    debug_assert!(bits == 0);
}

pub fn byte_decode<const D: usize>(bytes: &[u8]) -> Poly {
    debug_assert!(D > 0 && D <= 12);

    let mut out = [0i16; MlKem768::N];
    let mut acc: u64 = 0;
    let mut bits: usize = 0;
    let mut idx: usize = 0;
    let mask: u64 = (1u64 << D) - 1;

    for &b in bytes.iter() {
        acc |= (b as u64) << bits;
        bits += 8;

        while bits >= D {
            debug_assert!(idx < MlKem768::N);
            let mut val = (acc & mask) as i16;
            
            // FIPS 203 Algorithm 6 requires reducing mod Q if D=12
            if D == 12 {
                val %= MlKem768::Q as i16;
            }
            
            out[idx] = val;
            acc >>= D;
            bits -= D;
            idx += 1;
        }
    }

    debug_assert!(bits == 0 && idx == MlKem768::N);
    Poly(out)
}

/// Compress_d / Decompress_d for coefficient vectors
pub fn compress<const D: usize>(p: &Poly) -> [u16; MlKem768::N] {
    debug_assert!(D > 0 && D <= 12);

    let mut out = [0u16; MlKem768::N];
    let q = MlKem768::Q as i32;
    let scale = 1i32 << D;
    let mask = (scale as u16) - 1;
    let offset = q >> 1;

    for (i, &coef) in p.0.iter().enumerate() {
        // Map coefficient to [0, q) then scale to D bits with rounding.
        let mut x = coef as i32 % q;
        if x < 0 { x += q; }
        let t = ((x * scale + offset) / q) as u16;
        out[i] = t & mask;
    }

    out
}

pub fn decompress<const D: usize>(c: &[u16; MlKem768::N]) -> Poly {
    debug_assert!(D > 0 && D <= 12);

    let mut out = [0i16; MlKem768::N];
    let q = MlKem768::Q as i32;
    let rounding = 1i32 << (D - 1);
    let shift = D as i32;

    for (i, &t) in c.iter().enumerate() {
        // Inverse of compress: scale back to [0, q) with rounding.
        let val = ((t as i32 * q + rounding) >> shift) as i16;
        out[i] = val;
    }

    Poly(out)
}

/// Helpers for packing ciphertext (du,dv parts)
pub fn pack_ciphertext(u: &[[u16; MlKem768::N]; MlKem768::K], v: &[u16; MlKem768::N], out: &mut [u8; MlKem768::CT_BYTES]) {
    let mut acc: u32 = 0;
    let mut bits: usize = 0;
    let mut idx: usize = 0;
    let du_mask: u32 = (1u32 << MlKem768::DU) - 1;
    let dv_mask: u32 = (1u32 << MlKem768::DV) - 1;

    // pack u (K polynomials with du bits each coefficient)
    for poly in u.iter() {
        for &coef in poly.iter() {
            acc |= ((coef as u32) & du_mask) << bits;
            bits += MlKem768::DU;

            while bits >= 8 {
                out[idx] = acc as u8;
                acc >>= 8;
                bits -= 8;
                idx += 1;
            }
        }
    }

    // pack v (single polynomial with dv bits each coefficient)
    for &coef in v.iter() {
        acc |= ((coef as u32) & dv_mask) << bits;
        bits += MlKem768::DV;

        while bits >= 8 {
            out[idx] = acc as u8;
            acc >>= 8;
            bits -= 8;
            idx += 1;
        }
    }

    debug_assert!(bits == 0 && idx == MlKem768::CT_BYTES);
}

pub fn unpack_ciphertext(ct: &[u8; MlKem768::CT_BYTES]) -> ([[u16; MlKem768::N]; MlKem768::K], [u16; MlKem768::N]) {
    let mut u = [[0u16; MlKem768::N]; MlKem768::K];
    let mut v = [0u16; MlKem768::N];

    let mut acc: u32 = 0;
    let mut bits: usize = 0;
    let mut idx: usize = 0;
    let du_mask: u32 = (1u32 << MlKem768::DU) - 1;
    let dv_mask: u32 = (1u32 << MlKem768::DV) - 1;

    // unpack u
    for poly in 0..MlKem768::K {
        for coef in 0..MlKem768::N {
            while bits < MlKem768::DU {
                acc |= (ct[idx] as u32) << bits;
                bits += 8;
                idx += 1;
            }
            u[poly][coef] = (acc & du_mask) as u16;
            acc >>= MlKem768::DU;
            bits -= MlKem768::DU;
        }
    }

    // unpack v
    for coef in 0..MlKem768::N {
        while bits < MlKem768::DV {
            acc |= (ct[idx] as u32) << bits;
            bits += 8;
            idx += 1;
        }
        v[coef] = (acc & dv_mask) as u16;
        acc >>= MlKem768::DV;
        bits -= MlKem768::DV;
    }

    debug_assert!(bits == 0 && idx == MlKem768::CT_BYTES);
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_byte_encode_decode_12() {
        let mut p = Poly::zero();
        for i in 0..MlKem768::N {
            p.0[i] = (i * 13) as i16 % MlKem768::Q as i16;
        }
        let mut encoded = [0u8; MlKem768::POLY_BYTES_12];
        byte_encode::<12>(&p, &mut encoded);
        let decoded = byte_decode::<12>(&encoded);
        for i in 0..MlKem768::N {
            assert_eq!(p.0[i], decoded.0[i], "Mismatch at index {}", i);
        }
    }

    #[test]
    fn pack_unpacked_ciphertext_matches_original() {
        let mut rng = rand::thread_rng();
        let mut u = [[0u16; MlKem768::N]; MlKem768::K];
        let mut v = [0u16; MlKem768::N];

        for _ in 0..4 {
            for poly in u.iter_mut() {
                for coef in poly.iter_mut() {
                    *coef = rng.gen_range(0..(1u16 << MlKem768::DU));
                }
            }
            for coef in v.iter_mut() {
                *coef = rng.gen_range(0..(1u16 << MlKem768::DV));
            }

            let mut ct = [0u8; MlKem768::CT_BYTES];
            pack_ciphertext(&u, &v, &mut ct);
            let (u2, v2) = unpack_ciphertext(&ct);
            assert_eq!(u, u2);
            assert_eq!(v, v2);
        }
    }

    #[test]
    fn pack_handles_boundary_values() {
        let mut u = [[0u16; MlKem768::N]; MlKem768::K];
        let mut v = [0u16; MlKem768::N];

        for (i, poly) in u.iter_mut().enumerate() {
            for (j, coef) in poly.iter_mut().enumerate() {
                *coef = ((i + j) as u16) & ((1u16 << MlKem768::DU) - 1);
            }
        }
        for (i, coef) in v.iter_mut().enumerate() {
            *coef = (i as u16) & ((1u16 << MlKem768::DV) - 1);
        }

        let mut ct = [0u8; MlKem768::CT_BYTES];
        pack_ciphertext(&u, &v, &mut ct);
        let (u2, v2) = unpack_ciphertext(&ct);
        assert_eq!(u, u2);
        assert_eq!(v, v2);
    }
}
