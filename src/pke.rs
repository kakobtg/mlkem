use crate::params::MlKem768;
use crate::{Ek, Ct, MlKemError};
use crate::poly::{Poly, PolyVec};
use crate::{sample, encode, ntt};
use crate::reduce;

pub const SK_PKE_BYTES: usize = MlKem768::DK_BYTES - MlKem768::EK_BYTES - 64;

pub fn keygen(rho: &[u8; 32], sigma: &[u8; 32]) -> (Ek, [u8; SK_PKE_BYTES]) {
    let mut ek = [0u8; MlKem768::EK_BYTES];
    let mut sk = [0u8; SK_PKE_BYTES];

    // sample secret s and error e using sigma
    let mut s = [Poly::zero(); MlKem768::K];
    let mut e = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        s[i] = sample::sample_poly_cbd_eta(sigma, i as u8, MlKem768::ETA1);
        e[i] = sample::sample_poly_cbd_eta(sigma, (MlKem768::K + i) as u8, MlKem768::ETA1);
    }

    // NTT(s)
    let mut s_hat = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        s_hat[i] = ntt::ntt(s[i]);
        for coef in s_hat[i].0.iter_mut() {
            *coef = reduce::mod_q(*coef as i32);
        }
    }

    // build A from rho
    let mut a_hat = [[Poly::zero(); MlKem768::K]; MlKem768::K];
    for i in 0..MlKem768::K {
        for j in 0..MlKem768::K {
            a_hat[i][j] = sample::sample_ntt(rho, i as u8, j as u8);
        }
    }

    // compute t = invNTT(A_hat * s_hat) + e
    let mut t = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        let mut acc = ntt::mul_ntt(&a_hat[i][0], &s_hat[0]);
        for j in 1..MlKem768::K {
            acc = acc.add(&ntt::mul_ntt(&a_hat[i][j], &s_hat[j]));
        }
        t[i] = ntt::inv_ntt(acc).add(&e[i]);
        for coef in t[i].0.iter_mut() {
            *coef = reduce::mod_q(*coef as i32);
        }
    }

    // encode ek = t || rho
    let mut off = 0;
    for poly in t.iter() {
        encode::byte_encode::<12>(poly, &mut ek[off..off + MlKem768::POLY_BYTES_12]);
        off += MlKem768::POLY_BYTES_12;
    }
    ek[off..off + 32].copy_from_slice(rho);

    // encode sk = s_hat
    let mut sk_off = 0;
    for poly in s_hat.iter() {
        encode::byte_encode::<12>(poly, &mut sk[sk_off..sk_off + MlKem768::POLY_BYTES_12]);
        sk_off += MlKem768::POLY_BYTES_12;
    }

    (ek, sk)
}

/// Parse ek into (t_hat vector, rho)
pub fn parse_ek(_ek: &Ek) -> Result<(PolyVec, [u8; 32]), MlKemError> {
    let mut t_vec = [Poly::zero(); MlKem768::K];

    let mut offset = 0;
    for poly in t_vec.iter_mut() {
        let end = offset + MlKem768::POLY_BYTES_12;
        *poly = encode::byte_decode::<12>(&_ek[offset..end]);
        offset = end;
    }

    let mut rho = [0u8; 32];
    rho.copy_from_slice(&_ek[offset..offset + 32]);

    Ok((t_vec, rho))
}

/// PKE.Encrypt (internal): takes message poly and coins
pub fn encrypt(_ek: &Ek, _m: &[u8; 32], _coins: &[u8; 32]) -> Result<Ct, MlKemError> {
    // 1) parse ek -> (t_hat, rho)
    let (t_vec, rho) = parse_ek(_ek)?;

    // precompute NTT(t)
    let mut t_hat_vec = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        t_hat_vec[i] = ntt::ntt(t_vec[i]);
        for coef in t_hat_vec[i].0.iter_mut() {
            *coef = reduce::mod_q(*coef as i32);
        }
    }

    // 2) generate A_hat from rho via SampleNTT
    let mut a_hat = [[Poly::zero(); MlKem768::K]; MlKem768::K];
    for i in 0..MlKem768::K {
        for j in 0..MlKem768::K {
            a_hat[i][j] = sample::sample_ntt(&rho, i as u8, j as u8);
        }
    }

    let mut r = [Poly::zero(); MlKem768::K];
    let mut e1 = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        r[i] = sample::sample_poly_cbd_eta(_coins, i as u8, MlKem768::ETA1);
        e1[i] = sample::sample_poly_cbd_eta(_coins, (MlKem768::K + i) as u8, MlKem768::ETA2);
    }
    let e2 = sample::sample_poly_cbd_eta(_coins, (2 * MlKem768::K) as u8, MlKem768::ETA2);

    // 4) compute u = invNTT(A_hat^T * NTT(r)) + e1
    let mut r_hat = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        r_hat[i] = ntt::ntt(r[i]);
    }

    let mut u = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        let mut acc = ntt::mul_ntt(&a_hat[0][i], &r_hat[0]);
        for j in 1..MlKem768::K {
            acc = acc.add(&ntt::mul_ntt(&a_hat[j][i], &r_hat[j]));
        }
        u[i] = ntt::inv_ntt(acc).add(&e1[i]);
        for coef in u[i].0.iter_mut() {
            *coef = reduce::mod_q(*coef as i32);
        }
    }

    // 5) v = invNTT(t_hat^T * NTT(r)) + e2 + m
    let mut v_acc = ntt::mul_ntt(&t_hat_vec[0], &r_hat[0]);
    for j in 1..MlKem768::K {
        v_acc = v_acc.add(&ntt::mul_ntt(&t_hat_vec[j], &r_hat[j]));
    }
    let mut v = ntt::inv_ntt(v_acc).add(&e2);
    for coef in v.0.iter_mut() {
        *coef = reduce::mod_q(*coef as i32);
    }

    // Embed message bits as polynomial with coefficients in {0, (q+1)/2}
    let mut m_poly = [0i16; MlKem768::N];
    let msg_val = ((MlKem768::Q + 1) / 2) as i16;
    for (byte_idx, byte) in _m.iter().enumerate() {
        let base = byte_idx * 8;
        for bit in 0..8 {
            let mask = (byte >> bit) & 1;
            m_poly[base + bit] = if mask == 1 { msg_val } else { 0 };
        }
    }
    v = v.add(&Poly(m_poly));

    // 6) compress+pack with du,dv
    let mut u_comp = [[0u16; MlKem768::N]; MlKem768::K];
    for i in 0..MlKem768::K {
        u_comp[i] = encode::compress::<{ MlKem768::DU }>(&u[i]);
    }
    let v_comp = encode::compress::<{ MlKem768::DV }>(&v);

    let mut ct = [0u8; MlKem768::CT_BYTES];
    encode::pack_ciphertext(&u_comp, &v_comp, &mut ct);

    Ok(ct)
}

/// PKE.Decrypt (internal)
pub fn decrypt(_dk: &[u8], _ct: &Ct) -> Result<[u8; 32], MlKemError> {
    // unpack ciphertext
    let (u_comp, v_comp) = encode::unpack_ciphertext(_ct);

    // decode secret key (s_hat polynomials)
    let mut s_hat_vec = [Poly::zero(); MlKem768::K];
    let mut offset = 0;
    for poly in s_hat_vec.iter_mut() {
        let end = offset + MlKem768::POLY_BYTES_12;
        *poly = encode::byte_decode::<12>(&_dk[offset..end]);
        offset = end;
    }

    // compute v - invNTT(s_hat^T * NTT(u))
    let mut u_hat = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        let u_poly = encode::decompress::<{ MlKem768::DU }>(&u_comp[i]);
        u_hat[i] = ntt::ntt(u_poly);
    }

    let mut acc = ntt::mul_ntt(&s_hat_vec[0], &u_hat[0]);
    for j in 1..MlKem768::K {
        acc = acc.add(&ntt::mul_ntt(&s_hat_vec[j], &u_hat[j]));
    }
    let v_poly = encode::decompress::<{ MlKem768::DV }>(&v_comp);
    let m_poly = v_poly.sub(&ntt::inv_ntt(acc));

    // slice out message bits
    let mut m = [0u8; 32];
    for (i, &coef) in m_poly.0.iter().enumerate() {
        let val = reduce::mod_q(coef as i32) as i32;
        let t = ((val << 1) + (MlKem768::Q as i32 / 2)) / (MlKem768::Q as i32);
        let bit = (t & 1) as u8;
        m[i >> 3] |= bit << (i & 7);
    }

    Ok(m)
}