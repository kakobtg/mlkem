//! One comprehensive, end-to-end pipeline test for the whole `mlkem` crate.
//!
//! Rather than many small unit tests, this drives the library the way a real
//! caller would: generate a keypair, ship keys/ciphertexts across a simulated
//! wire, encapsulate/decapsulate, and actually use the resulting shared
//! secret to run an AEAD channel — while also covering the adversarial paths
//! (eavesdropper, tampered ciphertext, wrong key) and cross-checking the
//! deterministic internal API that backs the NIST KAT harness.

use mlkem::{decaps, encaps, keygen, params::MlKem768, Ct, Dk, Ek, Ss};
use rand::rngs::OsRng;
use std::collections::HashSet;

/// Minimal AES-256-GCM channel, keyed directly by an ML-KEM shared secret.
/// Mirrors the `three_party_exchange` example: proof that the derived secret
/// is actually usable as a symmetric key, not just a byte blob that compares
/// equal.
mod aes_channel {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Key, Nonce,
    };

    pub struct SecureChannel {
        cipher: Aes256Gcm,
    }

    impl SecureChannel {
        pub fn new(shared_secret: &[u8; 32]) -> Self {
            let key: &Key<Aes256Gcm> = shared_secret.into();
            Self {
                cipher: Aes256Gcm::new(key),
            }
        }

        pub fn encrypt(&self, plaintext: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>, &'static str> {
            self.cipher
                .encrypt(Nonce::from_slice(nonce), plaintext)
                .map_err(|_| "encryption failed")
        }

        pub fn decrypt(&self, ciphertext: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>, &'static str> {
            self.cipher
                .decrypt(Nonce::from_slice(nonce), ciphertext)
                .map_err(|_| "decryption failed")
        }
    }
}

/// Round-trips a fixed-size byte array through hex encode/decode, simulating
/// serialization across a wire (network, disk, etc.) between pipeline stages.
fn wire_roundtrip<const N: usize>(bytes: &[u8; N]) -> [u8; N] {
    let encoded = hex::encode(bytes);
    let decoded = hex::decode(&encoded).expect("hex decode failed");
    decoded
        .try_into()
        .expect("wire transfer changed the payload length")
}

/// Cross-checks the deterministic internal API (used by `tests/nist_kats.rs`)
/// for self-consistency: same seeds must always produce the same keys,
/// ciphertext and shared secret, and the internal decaps must recover
/// exactly what internal encaps produced. Only compiled/run when the
/// `test-utils` feature is enabled, since that's what gates `mlkem::internal`.
#[cfg(feature = "test-utils")]
fn check_deterministic_internal_pipeline() {
    use mlkem::internal::{decaps_internal, encaps_internal, keygen_internal};

    let d = [7u8; 32];
    let z = [9u8; 32];
    let m = [3u8; 32];

    let kp_a = keygen_internal(&d, &z);
    let kp_b = keygen_internal(&d, &z);
    assert_eq!(kp_a.ek, kp_b.ek, "deterministic keygen must reproduce ek for the same seeds");
    assert_eq!(kp_a.dk, kp_b.dk, "deterministic keygen must reproduce dk for the same seeds");

    let (ct_a, ss_a) = encaps_internal(&m, &kp_a.ek).expect("encaps_internal failed");
    let (ct_b, ss_b) = encaps_internal(&m, &kp_b.ek).expect("encaps_internal failed");
    assert_eq!(ct_a, ct_b, "deterministic encaps must reproduce ct for the same inputs");
    assert_eq!(ss_a, ss_b, "deterministic encaps must reproduce ss for the same inputs");

    let ss_dec = decaps_internal(&kp_a.dk, &ct_a).expect("decaps_internal failed");
    assert_eq!(ss_dec, ss_a, "internal pipeline must recover the same shared secret it encapsulated");
}

#[cfg(not(feature = "test-utils"))]
fn check_deterministic_internal_pipeline() {
    // `mlkem::internal` only exists behind the `test-utils` feature; run
    // `cargo test -p mlkem --features test-utils` to also exercise it here
    // (it's exercised unconditionally by `tests/nist_kats.rs`).
}

#[test]
fn ml_kem_768_end_to_end_pipeline() {
    // Parameter sanity: fixed-size types must match the FIPS 203 ML-KEM-768 sizes.
    assert_eq!(std::mem::size_of::<Ek>(), MlKem768::EK_BYTES);
    assert_eq!(std::mem::size_of::<Dk>(), MlKem768::DK_BYTES);
    assert_eq!(std::mem::size_of::<Ct>(), MlKem768::CT_BYTES);
    assert_eq!(std::mem::size_of::<Ss>(), MlKem768::SS_BYTES);
    assert_eq!(MlKem768::EK_BYTES, 1184);
    assert_eq!(MlKem768::DK_BYTES, 2400);
    assert_eq!(MlKem768::CT_BYTES, 1088);
    assert_eq!(MlKem768::SS_BYTES, 32);

    let mut rng = OsRng;

    // Alice generates a long-term keypair.
    let alice = keygen(&mut rng);

    // Alice's public key crosses the "wire" to Bob.
    let ek_on_wire = wire_roundtrip(&alice.ek);
    assert_eq!(ek_on_wire, alice.ek, "encapsulation key must survive serialization untouched");

    // Bob encapsulates a shared secret against Alice's public key.
    let (ct_bob, ss_bob) = encaps(&mut rng, &ek_on_wire).expect("Bob's encapsulation failed");
    assert_ne!(ss_bob, [0u8; 32], "shared secret must not be degenerate/all-zero");

    // The ciphertext crosses the "wire" back to Alice.
    let ct_on_wire = wire_roundtrip(&ct_bob);
    assert_eq!(ct_on_wire, ct_bob, "ciphertext must survive serialization untouched");

    // Alice decapsulates and must recover exactly Bob's shared secret.
    let ss_alice = decaps(&alice.dk, &ct_on_wire).expect("Alice's decapsulation failed");
    assert_eq!(ss_alice, ss_bob, "Alice and Bob must agree on the shared secret");

    // Put the shared secret to real use: an AES-256-GCM channel.
    let alice_channel = aes_channel::SecureChannel::new(&ss_alice);
    let bob_channel = aes_channel::SecureChannel::new(&ss_bob);
    let nonce = *b"pipeline-nc!"; // 12 bytes, required by AES-GCM
    let plaintext = b"the eagle lands at midnight";

    let ciphertext = bob_channel
        .encrypt(plaintext, &nonce)
        .expect("Bob must be able to encrypt with the shared secret");
    let recovered = alice_channel
        .decrypt(&ciphertext, &nonce)
        .expect("Alice must be able to decrypt Bob's message with the shared secret");
    assert_eq!(recovered, plaintext, "round-tripped plaintext must match exactly");

    // Eve: a passive eavesdropper with her own keypair must not derive the secret.
    let eve = keygen(&mut rng);
    let eve_ss = decaps(&eve.dk, &ct_on_wire)
        .expect("decaps must not error for a mismatched key (implicit rejection)");
    assert_ne!(eve_ss, ss_bob, "Eve must not be able to derive the real shared secret");
    let eve_channel = aes_channel::SecureChannel::new(&eve_ss);
    assert!(
        eve_channel.decrypt(&ciphertext, &nonce).is_err(),
        "Eve's bogus secret must not open Alice and Bob's AEAD channel"
    );

    // Tamper detection / implicit rejection on a corrupted ciphertext.
    let mut tampered_ct = ct_bob;
    tampered_ct[0] ^= 0x01;
    let ss_tampered =
        decaps(&alice.dk, &tampered_ct).expect("decaps must not error on a tampered ciphertext");
    assert_ne!(ss_tampered, ss_bob, "implicit rejection must change the derived secret on tampering");
    let tampered_channel = aes_channel::SecureChannel::new(&ss_tampered);
    assert!(
        tampered_channel.decrypt(&ciphertext, &nonce).is_err(),
        "a tampering-triggered secret must not open the original channel"
    );

    // Implicit rejection must be a deterministic function of (dk, ct), not noise.
    let ss_tampered_again =
        decaps(&alice.dk, &tampered_ct).expect("decaps must not error on a tampered ciphertext");
    assert_eq!(
        ss_tampered, ss_tampered_again,
        "implicit rejection output must be deterministic for the same dk/ct pair"
    );

    // Wrong keypair decapsulating the *original* ciphertext must also disagree.
    let ss_wrong_key = decaps(&eve.dk, &ct_bob).expect("decaps must not error for a wrong keypair");
    assert_ne!(ss_wrong_key, ss_bob, "decapsulation with the wrong secret key must not match");

    // Repeat across independent sessions: fresh randomness must yield fresh
    // keys, ciphertexts and secrets every time.
    let mut seen_secrets = HashSet::new();
    seen_secrets.insert(ss_bob);
    for _ in 0..5 {
        let kp = keygen(&mut rng);
        let (ct, ss_enc) = encaps(&mut rng, &kp.ek).expect("encapsulation failed in session loop");
        let ss_dec = decaps(&kp.dk, &ct).expect("decapsulation failed in session loop");
        assert_eq!(ss_enc, ss_dec, "sender/receiver secrets must match in every independent session");
        assert!(
            seen_secrets.insert(ss_enc),
            "shared secrets must not repeat across independent sessions"
        );
    }

    // Cross-check the deterministic internal API backing the NIST KAT harness.
    check_deterministic_internal_pipeline();
}
