use zeroize::Zeroize;

use crate::params::MlKem768;
use crate::pke::SK_PKE_BYTES;
use crate::util;
use crate::{ct, hash, pke};
use crate::{Ct, Dk, Ek, KeyPair, MlKemError, Ss};

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
///
/// FIPS 203, Algorithm 16 (ML-KEM.KeyGen_internal), step 1: `(ekPKE, dkPKE) ←
/// K-PKE.KeyGen(d)` — the seed `d` is passed straight through to K-PKE; this
/// layer does no hashing of its own. (K-PKE.KeyGen performs its own
/// `G(d ‖ k)` internally — see `pke::keygen`.)
pub fn keygen_internal_768(d: &[u8; 32], z: &[u8; 32]) -> KeyPair {
    let (ek, sk_pke) = pke::keygen(d);

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
///
/// FIPS 203, Algorithm 17 (ML-KEM.Encaps_internal): `(K, r) ← G(m ‖ H(ek))`,
/// `c ← K-PKE.Encrypt(ek, m, r)`, return `(K, c)`. `K` is returned directly —
/// there is no further hashing of `K` with the ciphertext (that extra step
/// existed in the Kyber round-3 design but was dropped in the final FIPS 203).
pub fn encaps_internal_768(m: &[u8; 32], ek: &Ek) -> Result<(Ct, Ss), MlKemError> {
    let h_ek = hash::h_sha3_256(ek);

    let mut g_in = [0u8; 64];
    g_in[..32].copy_from_slice(m);
    g_in[32..].copy_from_slice(&h_ek);
    let g_out = hash::g_sha3_512(&g_in);
    let (k, r_rest) = util::split_at_32(&g_out, 0);
    let mut r = [0u8; 32];
    r.copy_from_slice(&r_rest[..32]);

    let ct = pke::encrypt(ek, m, &r)?;

    Ok((ct, k))
}

/// FIPS 203, Algorithm 18 (ML-KEM.Decaps_internal): the implicit-rejection
/// value is `K̄ ← J(z ‖ c)` — hashed together with the *raw* ciphertext `c`,
/// not `H(c)` (again, `H(c)` was the round-3 construction; FIPS 203 hashes
/// the ciphertext itself). The "good" branch returns `K'` directly, with no
/// extra hashing, matching `encaps_internal_768` above.
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

    let mut bad_in = [0u8; 32 + MlKem768::CT_BYTES];
    bad_in[..32].copy_from_slice(&z);
    bad_in[32..].copy_from_slice(ct_in);
    let k_bar = hash::j_shake256_32(&bad_in);

    let ss = ct::ct_select_bytes::<32>(valid, &k_prime, &k_bar);

    Ok(ss)
}
