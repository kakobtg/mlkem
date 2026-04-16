use sha3::{Digest, Sha3_256, Sha3_512};
use sha3::Shake128;
use sha3::Shake256;
use sha3::digest::{Update, ExtendableOutput, XofReader};

/// H = SHA3-256(x) -> 32 bytes
pub fn h_sha3_256(input: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    Update::update(&mut hasher, input);
    let out = hasher.finalize();
    let mut r = [0u8; 32];
    r.copy_from_slice(&out);
    r
}

/// G = SHA3-512(x) -> 64 bytes
pub fn g_sha3_512(input: &[u8]) -> [u8; 64] {
    let mut hasher = Sha3_512::new();
    Update::update(&mut hasher, input);
    let out = hasher.finalize();
    let mut r = [0u8; 64];
    r.copy_from_slice(&out);
    r
}

/// J = SHAKE256(x, 32 bytes)
pub fn j_shake256_32(input: &[u8]) -> [u8; 32] {
    let mut hasher = Shake256::default();
    Update::update(&mut hasher, input);
    let mut reader = hasher.finalize_xof();
    let mut out = [0u8; 32];
    reader.read(&mut out);
    out
}

/// PRFη(s, b): SHAKE256(s || b, outlen)
pub fn prf_shake256(seed: &[u8; 32], b: u8, out: &mut [u8]) {
    let mut hasher = Shake256::default();
    Update::update(&mut hasher, seed);
    Update::update(&mut hasher, &[b]);
    let mut reader = hasher.finalize_xof();
    reader.read(out);
}

/// XOF (for matrix generation etc): SHAKE128(seed || i || j, outlen)
#[allow(dead_code)]
pub fn xof_shake128(seed: &[u8; 32], i: u8, j: u8, out: &mut [u8]) {
    let mut hasher = Shake128::default();
    Update::update(&mut hasher, seed);
    Update::update(&mut hasher, &[i, j]);
    let mut reader = hasher.finalize_xof();
    reader.read(out);
}
