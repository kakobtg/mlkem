use mlkem::{keygen, encaps, decaps};
use rand::rngs::OsRng;

#[test]
fn kem_smoke() {
    let mut rng = OsRng;
    let kp = keygen(&mut rng);
    let (ct, ss1) = encaps(&mut rng, &kp.ek).unwrap();
    let ss2 = decaps(&kp.dk, &ct).unwrap();
    assert_eq!(ss1, ss2);
}


// This will fail until the KEM is fully implemented, but I will treat it as a "North Star" test