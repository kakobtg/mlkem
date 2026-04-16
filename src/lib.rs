//#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

pub mod error;
pub mod params;

mod kem;
mod pke;

mod hash;
mod encode;
mod reduce;
mod poly;
pub mod ntt;
mod sample;
mod ct;
mod util;

pub use error::MlKemError;
pub use params::MlKem768;

/// ML-KEM-768 fixed-size types (near real life: avoid Vec allocations)
pub type Ek = [u8; params::MlKem768::EK_BYTES];
pub type Dk = [u8; params::MlKem768::DK_BYTES];
pub type Ct = [u8; params::MlKem768::CT_BYTES];
pub type Ss = [u8; params::MlKem768::SS_BYTES];

/// Keypair container
#[derive(Clone)]
pub struct KeyPair {
    pub ek: Ek,
    pub dk: Dk,
}

/// Public API: randomized keygen (uses OS RNG or caller RNG via trait)
pub fn keygen<R: rand_core::RngCore + rand_core::CryptoRng>(rng: &mut R) -> KeyPair {
    kem::keygen_768(rng)
}

/// Public API: randomized encaps
pub fn encaps<R: rand_core::RngCore + rand_core::CryptoRng>(rng: &mut R, ek: &Ek) -> Result<(Ct, Ss), MlKemError> {
    kem::encaps_768(rng, ek)
}

/// Public API: decaps
pub fn decaps(dk: &Dk, ct: &Ct) -> Result<Ss, MlKemError> {
    kem::decaps_768(dk, ct)
}

/// Deterministic/internal interfaces (for KATs). Hidden behind feature flag.
#[cfg(feature = "test-utils")]
pub mod internal {
    use super::*;

    pub fn keygen_internal(d: &[u8; 32], z: &[u8; 32]) -> KeyPair {
        crate::kem::keygen_internal_768(d, z)
    }

    pub fn encaps_internal(m: &[u8; 32], ek: &Ek) -> Result<(Ct, Ss), MlKemError> {
        crate::kem::encaps_internal_768(m, ek)
    }

    pub fn decaps_internal(dk: &Dk, ct: &Ct) -> Result<Ss, MlKemError> {
        crate::kem::decaps_internal_768(dk, ct)
    }
}
