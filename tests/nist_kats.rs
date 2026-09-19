#![cfg(feature = "test-utils")]

//! Validates this crate against the official NIST ACVP test vectors for
//! ML-KEM-768, sourced from `usnistgov/ACVP-Server`
//! (`gen-val/json-files/ML-KEM-{keyGen,encapDecap}-FIPS203/internalProjection.json`,
//! ML-KEM-768 groups only — see `test_vectors/ML-KEM-768.json`).
//!
//! Covers three independent ACVP test groups, each exercised against the
//! deterministic `internal::` API:
//! - `keyGen` (AFT): `(d, z) -> (ek, dk)`
//! - `encapsulation` (AFT): `(m, ek) -> (c, k)`
//! - `decapsulation` (VAL): `(dk, c) -> k`, including "modified ciphertext"
//!   cases that exercise implicit rejection (decaps never errors — a bad
//!   ciphertext just yields a different, still-deterministic, `k`).

use mlkem::internal::{decaps_internal, encaps_internal, keygen_internal};

use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn decode_hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

fn field<const N: usize>(test: &Value, name: &str) -> [u8; N] {
    let hex = test[name].as_str().unwrap_or_else(|| panic!("missing field `{name}`"));
    decode_hex(hex)
        .try_into()
        .unwrap_or_else(|v: Vec<u8>| panic!("field `{name}` has {} bytes, expected {N}", v.len()))
}

#[test]
fn test_official_nist_kats() {
    let kat_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_vectors")
        .join("ML-KEM-768.json");

    // Gracefully skip if the vectors haven't been fetched, so this doesn't
    // break a build that hasn't run the (external, one-time) download step.
    if !kat_file.exists() {
        println!("ML-KEM-768 ACVP vectors not found at {kat_file:?}. Skipping test.");
        return;
    }

    let raw = fs::read_to_string(&kat_file).expect("failed to read KAT file");
    let doc: Value = serde_json::from_str(&raw).expect("failed to parse KAT JSON");

    let mut keys_tested = 0;
    for test in doc["keyGen"].as_array().expect("keyGen array") {
        let z: [u8; 32] = field(test, "z");
        let d: [u8; 32] = field(test, "d");
        let ek: mlkem::Ek = field(test, "ek");
        let dk: mlkem::Dk = field(test, "dk");

        let kp = keygen_internal(&d, &z);
        assert_eq!(kp.ek, ek, "keyGen tcId {}: ek mismatch", test["tcId"]);
        assert_eq!(kp.dk, dk, "keyGen tcId {}: dk mismatch", test["tcId"]);
        keys_tested += 1;
    }

    let mut encaps_tested = 0;
    for test in doc["encapsulation"].as_array().expect("encapsulation array") {
        let ek: mlkem::Ek = field(test, "ek");
        let m: [u8; 32] = field(test, "m");
        let c: mlkem::Ct = field(test, "c");
        let k: mlkem::Ss = field(test, "k");

        let (ct, ss) = encaps_internal(&m, &ek).expect("encaps_internal failed");
        assert_eq!(ct, c, "encapsulation tcId {}: ciphertext mismatch", test["tcId"]);
        assert_eq!(ss, k, "encapsulation tcId {}: shared secret mismatch", test["tcId"]);
        encaps_tested += 1;
    }

    let mut decaps_tested = 0;
    for test in doc["decapsulation"].as_array().expect("decapsulation array") {
        let dk: mlkem::Dk = field(test, "dk");
        let c: mlkem::Ct = field(test, "c");
        let k: mlkem::Ss = field(test, "k");
        let reason = test["reason"].as_str().unwrap_or("");

        let ss = decaps_internal(&dk, &c).expect("decaps_internal failed");
        assert_eq!(
            ss, k,
            "decapsulation tcId {} ({reason}): shared secret mismatch",
            test["tcId"]
        );
        decaps_tested += 1;
    }

    assert!(keys_tested > 0 && encaps_tested > 0 && decaps_tested > 0, "no KATs were executed");
    println!("Successfully passed official NIST ACVP ML-KEM-768 vectors!");
    println!("- KeyGen tested: {keys_tested}");
    println!("- Encaps tested: {encaps_tested}");
    println!("- Decaps tested: {decaps_tested}");
}
