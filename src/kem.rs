use zeroize::Zeroize;

use crate::params::MlKem768;
use crate::{KeyPair, Ek, Dk, Ct, Ss, MlKemError};
use crate::{hash, pke, ct};
use crate::pke::SK_PKE_BYTES;
use crate::util;

use rand_core::{CryptoRng, RngCore};

pub fn keygen_768<R: RngCore + CryptoRng>(rng: &mut R) -> KeyPair {
    let mut d = [0u8; 32];
    let mut z = [0u8; 32];
    rng.fill_bytes(&mut d);
    rng.fill_bytes(&mut z);
    let kp = keygen_internal_768(&d, &z);
    d.zeroize();
    z.zeroize();
    kp
}

pub fn encaps_768<R: RngCore + CryptoRng>(rng: &mut R, ek: &Ek) -> Result<(Ct, Ss), MlKemError> {
    let mut m = [0u8; 32];
    rng.fill_bytes(&mut m);
    let out = encaps_internal_768(&m, ek);
    m.zeroize();
    out
}

pub fn decaps_768(dk: &Dk, ct_in: &Ct) -> Result<Ss, MlKemError> {
    decaps_internal_768(dk, ct_in)
}

/// Deterministic keygen: input seeds (for KATs)
pub fn keygen_internal_768(d: &[u8; 32], z: &[u8; 32]) -> KeyPair {
    let g_out = hash::g_sha3_512(d);
    let (rho, sigma_rest) = util::split_at_32(&g_out, 0);
    let mut sigma = [0u8; 32];
    sigma.copy_from_slice(&sigma_rest[..32]);
    let (ek, sk_pke) = pke::keygen(&rho, &sigma);

    let h_ek = hash::h_sha3_256(&ek);

    let mut dk = [0u8; MlKem768::DK_BYTES];
    let mut off = 0;
    dk[off..off + SK_PKE_BYTES].copy_from_slice(&sk_pke);
    off += SK_PKE_BYTES;
    dk[off..off + MlKem768::EK_BYTES].copy_from_slice(&ek);
    off += MlKem768::EK_BYTES;
    dk[off..off + 32].copy_from_slice(&h_ek);
    off += 32;
    dk[off..off + 32].copy_from_slice(z);

    KeyPair { ek, dk }
}

/// Deterministic encaps: input m for KATs
pub fn encaps_internal_768(m: &[u8; 32], ek: &Ek) -> Result<(Ct, Ss), MlKemError> {
    let h_ek = hash::h_sha3_256(ek);

    let mut g_in = [0u8; 64];
    g_in[..32].copy_from_slice(m);
    g_in[32..].copy_from_slice(&h_ek);
    let g_out = hash::g_sha3_512(&g_in);
    let (k_prime, r_rest) = util::split_at_32(&g_out, 0);
    let mut r = [0u8; 32];
    r.copy_from_slice(&r_rest[..32]);

    let ct = pke::encrypt(ek, m, &r)?;
    let h_ct = hash::h_sha3_256(&ct);

    let mut j_in = [0u8; 64];
    j_in[..32].copy_from_slice(&k_prime);
    j_in[32..].copy_from_slice(&h_ct);
    let ss = hash::j_shake256_32(&j_in);

    Ok((ct, ss))
}

pub fn decaps_internal_768(dk: &Dk, ct_in: &Ct) -> Result<Ss, MlKemError> {
    // parse dk layout: sk_pke || ek || h_ek || z
    let mut off = 0;
    let mut sk_pke = [0u8; SK_PKE_BYTES];
    sk_pke.copy_from_slice(&dk[off..off + SK_PKE_BYTES]);
    off += SK_PKE_BYTES;

    let mut ek = [0u8; MlKem768::EK_BYTES];
    ek.copy_from_slice(&dk[off..off + MlKem768::EK_BYTES]);
    off += MlKem768::EK_BYTES;

    let h_ek = util::split_at_32(dk, off).0;
    off += 32;
    let z = util::split_at_32(dk, off).0;

    let m_prime = pke::decrypt(&sk_pke, ct_in)?;

    let mut g_in = [0u8; 64];
    g_in[..32].copy_from_slice(&m_prime);
    g_in[32..].copy_from_slice(&h_ek);
    let g_out = hash::g_sha3_512(&g_in);
    let (k_prime, r_rest) = util::split_at_32(&g_out, 0);
    let mut r_prime = [0u8; 32];
    r_prime.copy_from_slice(&r_rest[..32]);

    let ct_prime = pke::encrypt(&ek, &m_prime, &r_prime)?;
    let valid = ct::ct_eq(ct_in, &ct_prime);

    let h_ct = hash::h_sha3_256(ct_in);

    let mut good = [0u8; 64];
    good[..32].copy_from_slice(&k_prime);
    good[32..].copy_from_slice(&h_ct);

    let mut bad = [0u8; 64];
    bad[..32].copy_from_slice(&z);
    bad[32..].copy_from_slice(&h_ct);

    let chosen = ct::ct_select_bytes::<64>(valid, &good, &bad);
    let ss = hash::j_shake256_32(&chosen);

    Ok(ss)
}
