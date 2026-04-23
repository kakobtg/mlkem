use mlkem::{keygen, encaps, decaps};

#[test]
fn kem_smoke() {
    let mut rng = rand::thread_rng();

    let kp = keygen(&mut rng);
    let (ct, ss_enc) = encaps(&mut rng, &kp.ek).expect("Encapsulation failed");
    let ss_dec = decaps(&kp.dk, &ct).expect("Decapsulation failed");
    
    assert_eq!(ss_enc, ss_dec, "Shared secrets MUST match!");
}

#[test]
fn kem_implicit_rejection() {
    let mut rng = rand::thread_rng();
    let kp = keygen(&mut rng);
    let (mut ct, ss_enc) = encaps(&mut rng, &kp.ek).unwrap();

    // Corrupt ciphertext
    ct[0] ^= 1; 
    let ss_dec = decaps(&kp.dk, &ct).unwrap();
    assert_ne!(ss_enc, ss_dec, "Implicit rejection failed! Shared secrets matched despite corruption.");
}

#[test]
fn kem_key_pair_mismatch() {
    let mut rng = rand::thread_rng();
    let kp1 = keygen(&mut rng);
    let kp2 = keygen(&mut rng); // Different keypair

    let (ct, ss_enc) = encaps(&mut rng, &kp1.ek).unwrap();
    let ss_dec = decaps(&kp2.dk, &ct).unwrap();
    assert_ne!(ss_enc, ss_dec, "Decapsulation with wrong secret key matched!");
}