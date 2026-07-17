#![cfg(feature = "test-utils")]

// Note: Adjust these imports depending on how you expose your internal functions in `lib.rs`
use mlkem::internal::{keygen_internal, encaps_internal, decaps_internal};

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

/// A simple helper to decode hex strings without needing external crates like `hex`
fn decode_hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("Valid hex"))
        .collect()
}

#[test]
fn test_official_nist_kats() {
    let kat_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_vectors")
        .join("kat_MLKEM_768.rsp");

    // Gracefully skip the test if the user hasn't downloaded the KAT vectors yet,
    // avoiding breaking CI pipelines by default.
    if !kat_file.exists() {
        println!("NIST KAT file not found at {:?}. Skipping test.", kat_file);
        return;
    }

    let file = File::open(kat_file).expect("Failed to open KAT file");
    let reader = BufReader::new(file);

    let mut d = Vec::new();
    let mut z = Vec::new();
    let mut pk = Vec::new();
    let mut sk = Vec::new();
    let mut m = Vec::new();
    let mut ct = Vec::new();
    let mut count = 0;

    let mut keys_tested = 0;
    let mut encaps_tested = 0;
    let mut decaps_tested = 0;

    for line in reader.lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split(" = ").collect();
        if parts.len() != 2 { continue; }

        let key = parts[0];
        let val = parts[1];

        match key {
            "d" => d = decode_hex(val),
            "z" => z = decode_hex(val),
            "pk" => pk = decode_hex(val),
            "sk" => sk = decode_hex(val),
            "m" | "msg" => m = decode_hex(val),
            "ct" | "c" => ct = decode_hex(val),
            "ss" | "K" | "k" | "shared_secret" => {
                let ss = decode_hex(val);
                
                // 1. Test Deterministic KeyGen (only if d and z were present)
                if d.len() == 32 && z.len() == 32 {
                    let d_arr: [u8; 32] = d.as_slice().try_into().unwrap();
                    let z_arr: [u8; 32] = z.as_slice().try_into().unwrap();
                    let kp = keygen_internal(&d_arr, &z_arr);
                    assert_eq!(kp.ek.as_slice(), pk.as_slice(), "PK mismatch in KAT {}", count);
                    assert_eq!(kp.dk.as_slice(), sk.as_slice(), "SK mismatch in KAT {}", count);
                    keys_tested += 1;
                }

                // 2. Test Deterministic Encapsulation (only if m was present)
                if m.len() == 32 && pk.len() == 1184 {
                    let m_arr: [u8; 32] = m.as_slice().try_into().unwrap();
                    let pk_arr: mlkem::Ek = pk.as_slice().try_into().unwrap();
                    let (ct_out, ss_enc) = encaps_internal(&m_arr, &pk_arr).expect("Encaps failed");
                    assert_eq!(ct_out.as_slice(), ct.as_slice(), "Ciphertext mismatch in KAT {}", count);
                    assert_eq!(ss_enc.as_slice(), ss.as_slice(), "Encaps SS mismatch in KAT {}", count);
                    encaps_tested += 1;
                }

                // 3. Test Deterministic Decapsulation (requires sk and ct)
                if sk.len() == 2400 && ct.len() == 1088 {
                    let sk_arr: mlkem::Dk = sk.as_slice().try_into().unwrap();
                    let ct_arr: mlkem::Ct = ct.as_slice().try_into().unwrap();
                    let ss_dec = decaps_internal(&sk_arr, &ct_arr).expect("Decaps failed");
                    assert_eq!(ss_dec.as_slice(), ss.as_slice(), "Decaps SS mismatch in KAT {}", count);
                    decaps_tested += 1;
                }

                count += 1;
                d.clear(); z.clear(); pk.clear(); sk.clear(); m.clear(); ct.clear();
            }
            _ => {}
        }
    }

    assert!(decaps_tested > 0, "No KATs were executed! Check the .rsp file format.");
    println!("Successfully passed NIST ML-KEM-768 KATs!");
    println!("- KeyGen tested: {}", keys_tested);
    println!("- Encaps tested: {}", encaps_tested);
    println!("- Decaps tested: {}", decaps_tested);
}