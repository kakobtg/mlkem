//! A simple example demonstrating the full ML-KEM-768 round-trip.
//!
//! To run this example:
//! ```bash
//! cargo run --example kem_roundtrip
//! ```

use mlkem::{keygen, encaps, decaps};
use rand::rngs::OsRng;

fn main() {
    println!("Running ML-KEM-768 round-trip example...");

    // 1. Create a cryptographically secure random number generator.
    //    `OsRng` is a good choice for this on most platforms.
    let mut rng = OsRng;

    // 2. Generate a public/private keypair for the KEM.
    let keypair = keygen(&mut rng);
    println!("Keypair generated.");

    // 3. The "sender" encapsulates a shared secret using the public key.
    //    This produces a ciphertext (ct) and the sender's shared secret (ss1).
    let (ciphertext, shared_secret_sender) = encaps(&mut rng, &keypair.ek)
        .expect("Encapsulation failed");
    println!("Shared secret encapsulated.");

    // 4. The "receiver" decapsulates the ciphertext using their private key
    //    to derive the same shared secret (ss2).
    let shared_secret_receiver = decaps(&keypair.dk, &ciphertext)
        .expect("Decapsulation failed");
    println!("Ciphertext decapsulated.");

    // 5. Verify that both parties have the same shared secret.
    assert_eq!(shared_secret_sender, shared_secret_receiver, "Shared secrets do not match!");
    println!("Success!!!!!!! Shared secrets match.");
}