use crate::params::MlKem768;
use crate::poly::{Poly, PolyVec};
use crate::reduce;
use crate::{encode, ntt, sample};
use crate::{Ct, Ek, MlKemError};
use sha3::{Digest, Sha3_512};

pub const SK_PKE_BYTES: usize = MlKem768::DK_BYTES - MlKem768::EK_BYTES - 64;

/// Samples the public matrix `A_hat` (already in the NTT domain) from `rho`.
/// Shared by K-PKE.KeyGen and K-PKE.Encrypt (FIPS 203, Algorithm 13 step 4 /
/// Algorithm 14 step 3) — both build the exact same matrix from the same seed.
fn sample_matrix_a(rho: &[u8; 32]) -> [PolyVec; MlKem768::K] {
    let mut a_hat = [[Poly::zero(); MlKem768::K]; MlKem768::K];
    for i in 0..MlKem768::K {
        for j in 0..MlKem768::K {
            a_hat[i][j] = sample::sample_ntt(rho, i as u8, j as u8);
        }
    }
    a_hat
}

/// Samples a length-K vector of CBD-noise polynomials from `seed`, using
/// nonces `nonce_base, nonce_base+1, ..., nonce_base+K-1`. Used for the
/// secret/error vectors in both KeyGen and Encrypt.
fn sample_noise_vec(seed: &[u8; 32], nonce_base: u8, eta: usize) -> PolyVec {
    let mut v = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        v[i] = sample::sample_poly_cbd_eta(seed, nonce_base + i as u8, eta);
    }
    v
}

/// Applies the forward NTT, then reduces, to every polynomial in a vector.
fn ntt_and_reduce_vec(v: &PolyVec) -> PolyVec {
    let mut out = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        out[i] = ntt::ntt(v[i]).reduced();
    }
    out
}

/// Un-reduced NTT-domain dot product `sum_j a[j] ⊙ b[j]`; callers apply
/// `.reduced()` themselves since where that happens varies by call site.
fn dot_ntt_raw(a: &PolyVec, b: &PolyVec) -> Poly {
    let mut acc = ntt::mul_ntt(&a[0], &b[0]);
    for j in 1..MlKem768::K {
        acc = acc.add(&ntt::mul_ntt(&a[j], &b[j]));
    }
    acc
}

pub fn keygen(d: &[u8; 32]) -> (Ek, [u8; SK_PKE_BYTES]) {
    let mut ek = [0u8; MlKem768::EK_BYTES];
    let mut sk = [0u8; SK_PKE_BYTES];

    // (ρ, σ) ← G(d ‖ k); appending `k` domain-separates the hash across
    // ML-KEM parameter sets (FIPS 203, Algorithm 13, Step 1).
    let mut hasher = Sha3_512::new();
    hasher.update(d);
    hasher.update([MlKem768::K as u8]);
    let g_out = hasher.finalize();

    let rho: &[u8; 32] = g_out[0..32].try_into().unwrap();
    let sigma: &[u8; 32] = g_out[32..64].try_into().unwrap();

    let s = sample_noise_vec(sigma, 0, MlKem768::ETA1);
    let e = sample_noise_vec(sigma, MlKem768::K as u8, MlKem768::ETA1);
    let s_hat = ntt_and_reduce_vec(&s);
    let a_hat = sample_matrix_a(rho);

    // t_hat = A_hat . s_hat + NTT(e)
    let mut t_hat = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        let e_hat = ntt::ntt(e[i]);
        t_hat[i] = dot_ntt_raw(&a_hat[i], &s_hat).add(&e_hat).reduced();
    }

    // ek = t_hat || rho
    let mut off = 0;
    for poly in t_hat.iter() {
        encode::byte_encode::<12>(poly, &mut ek[off..off + MlKem768::POLY_BYTES_12]);
        off += MlKem768::POLY_BYTES_12;
    }
    ek[off..off + 32].copy_from_slice(rho);

    // sk = s_hat
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
    let (t_hat_vec, rho) = parse_ek(_ek)?;
    let a_hat = sample_matrix_a(&rho);

    let r = sample_noise_vec(_coins, 0, MlKem768::ETA1);
    let e1 = sample_noise_vec(_coins, MlKem768::K as u8, MlKem768::ETA2);
    let e2 = sample::sample_poly_cbd_eta(_coins, (2 * MlKem768::K) as u8, MlKem768::ETA2);
    let r_hat = ntt_and_reduce_vec(&r);

    // u = invNTT(A_hat^T . NTT(r)) + e1
    let mut u = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        // column i of A_hat^T is row i of A_hat: {a_hat[j][i] : j in 0..K}
        let mut a_col_i = [Poly::zero(); MlKem768::K];
        for j in 0..MlKem768::K {
            a_col_i[j] = a_hat[j][i];
        }
        let acc = dot_ntt_raw(&a_col_i, &r_hat).reduced();
        u[i] = ntt::inv_ntt(acc).add(&e1[i]).reduced();
    }

    // v = invNTT(t_hat^T . NTT(r)) + e2 + m, m encoded as {0, (q+1)/2} per bit
    let v_acc = dot_ntt_raw(&t_hat_vec, &r_hat).reduced();
    let v_pre_msg = ntt::inv_ntt(v_acc).add(&e2).reduced();

    let mut m_poly = [0i16; MlKem768::N];
    let msg_val = ((MlKem768::Q + 1) / 2) as i16;
    for (byte_idx, byte) in _m.iter().enumerate() {
        let base = byte_idx * 8;
        for bit in 0..8 {
            let mask = (byte >> bit) & 1;
            m_poly[base + bit] = if mask == 1 { msg_val } else { 0 };
        }
    }
    let v = v_pre_msg.add(&Poly(m_poly));

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
    let (u_comp, v_comp) = encode::unpack_ciphertext(_ct);

    let mut s_hat_vec = [Poly::zero(); MlKem768::K];
    let mut offset = 0;
    for poly in s_hat_vec.iter_mut() {
        let end = offset + MlKem768::POLY_BYTES_12;
        *poly = encode::byte_decode::<12>(&_dk[offset..end]);
        offset = end;
    }

    // m = v - invNTT(s_hat^T . NTT(u))
    let mut u_polys = [Poly::zero(); MlKem768::K];
    for i in 0..MlKem768::K {
        u_polys[i] = encode::decompress::<{ MlKem768::DU }>(&u_comp[i]);
    }
    let u_hat = ntt_and_reduce_vec(&u_polys);

    let acc = dot_ntt_raw(&s_hat_vec, &u_hat).reduced();
    let v_poly = encode::decompress::<{ MlKem768::DV }>(&v_comp);
    let m_poly = v_poly.sub(&ntt::inv_ntt(acc));

    let mut m = [0u8; 32];
    for (i, &coef) in m_poly.0.iter().enumerate() {
        let val = reduce::mod_q(coef as i32) as i32;
        let t = ((val << 1) + (MlKem768::Q as i32 / 2)) / (MlKem768::Q as i32);
        let bit = (t & 1) as u8;
        m[i >> 3] |= bit << (i & 7);
    }

    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn test_pke_roundtrip_isolated() {
        let mut rng = rand::thread_rng();
        let mut d = [0u8; 32];
        let mut z = [0u8; 32];
        let mut coins = [0u8; 32];
        let mut msg = [0u8; 32];

        rng.fill_bytes(&mut d);
        rng.fill_bytes(&mut z);
        rng.fill_bytes(&mut coins);
        rng.fill_bytes(&mut msg);

        let (ek, dk) = keygen(&d);
        let ct = encrypt(&ek, &msg, &coins).unwrap();
        let dec_msg = decrypt(&dk, &ct).unwrap();

        assert_eq!(msg, dec_msg, "Core PKE Encrypt/Decrypt roundtrip failed!");
    }
}
