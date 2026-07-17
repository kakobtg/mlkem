//! An example demonstrating a three-party key exchange scenario.
//!
//! SCENARIO:
//! 1. ALICE wants to establish a secure channel with BOB.
//! 2. EVE is a passive eavesdropper listening to all communication.
//! 3. ALICE generates a keypair and sends her public key to BOB.
//! 4. BOB uses the public key to create a shared secret and a ciphertext.
//!    He sends the ciphertext back to ALICE.
//! 5. EVE intercepts the public key and the ciphertext but cannot derive the secret.
//! 6. ALICE uses her private key to derive the same shared secret.
//! 7. ALICE and BOB now share a secret key that EVE does not know, which they
//!    can use for symmetric encryption (e.g., AES).
//!
//! To run this example, you must first add `aes-gcm` to the `[dev-dependencies]`
//! section of the `mlkem` crate's `Cargo.toml` file.
//!
//! Then run:
//! `cargo run --example three_party_exchange -p mlkem`

use mlkem::{keygen, encaps, decaps};
use rand::rngs::OsRng;

/// A simple module to represent the symmetric AES channel.
mod aes_channel {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Key, Nonce,
    };

    /// A simple wrapper for an AES-256-GCM channel.
    pub struct SecureChannel {
        cipher: Aes256Gcm,
    }

    impl SecureChannel {
        /// Creates a new secure channel initialized with the shared secret from the KEM.
        pub fn new(shared_secret: &[u8; 32]) -> Self {
            let key: &Key<Aes256Gcm> = shared_secret.into();
            Self {
                cipher: Aes256Gcm::new(key),
            }
        }

        /// Encrypts a message. A unique nonce must be used for each message.
        pub fn encrypt(&self, plaintext: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>, &'static str> {
            let nonce = Nonce::from_slice(nonce);
            self.cipher
                .encrypt(nonce, plaintext)
                .map_err(|_| "Encryption failed")
        }

        /// Decrypts a message. The same nonce used for encryption must be provided.
        pub fn decrypt(&self, ciphertext: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>, &'static str> {
            let nonce = Nonce::from_slice(nonce);
            self.cipher
                .decrypt(nonce, ciphertext)
                .map_err(|_| "Decryption failed! The message may have been tampered with.")
        }
    }
}


fn main() {
    println!("--- Setting up the three-party key exchange scenario ---");
    let mut rng = OsRng;

    // --- 1. Alice generates her keypair ---
    println!("[ALICE] Generating ML-KEM keypair...");
    let alice_keys = keygen(&mut rng);
    println!("[ALICE] Public key sent over the wire.");
    println!("\n------------------------------------------------------\n");

    // --- 2. Bob receives Alice's public key and encapsulates a secret ---
    println!("[BOB]   Received Alice's public key.");
    println!("[BOB]   Encapsulating a shared secret...");
    let (bob_ciphertext, bob_shared_secret) = encaps(&mut rng, &alice_keys.ek)
        .expect("Bob failed to encapsulate");
    println!("[BOB]   Ciphertext sent over the wire.");
    println!("\n------------------------------------------------------\n");

    // --- 3. Eve, the eavesdropper, intercepts the public traffic ---
    println!("[EVE]   Intercepted Alice's public key and Bob's ciphertext.");
    // Eve has no choice but to generate her own keys to try and decapsulate.
    let eve_keys = keygen(&mut rng);
    println!("[EVE]   Trying to decapsulate with her own private key...");
    // This will "succeed" but produce a garbage secret due to implicit rejection.
    let eve_fake_secret = decaps(&eve_keys.dk, &bob_ciphertext)
        .expect("Eve's decapsulation should not error");
    println!("[EVE]   Derived a fake secret. Is it the right one?");
    println!("\n------------------------------------------------------\n");

    // --- 4. Alice receives Bob's ciphertext and decapsulates it ---
    println!("[ALICE] Received Bob's ciphertext.");
    println!("[ALICE] Decapsulating with her private key...");
    let alice_shared_secret = decaps(&alice_keys.dk, &bob_ciphertext)
        .expect("Alice failed to decapsulate");
    println!("[ALICE] Successfully derived the shared secret.");
    println!("\n------------------------------------------------------\n");

    // --- 5. Verification ---
    println!("--- VERIFICATION ---");
    assert_eq!(
        alice_shared_secret, bob_shared_secret,
        "FATAL: Alice and Bob's secrets do not match!"
    );
    println!("SUCCESS: Alice and Bob have the same shared secret.");

    assert_ne!(
        bob_shared_secret, eve_fake_secret,
        "FATAL: Eve managed to guess the secret!"
    );
    println!("SUCCESS: Eve's secret does NOT match.");
    println!("\n--- DEMONSTRATING SECURE COMMUNICATION ---");

    // --- 6. Alice and Bob use the key for AES encryption ---
    // Both can now independently create a secure channel with the shared secret.
    let alice_channel = aes_channel::SecureChannel::new(&alice_shared_secret);
    let bob_channel = aes_channel::SecureChannel::new(&bob_shared_secret);

    // A 96-bit (12-byte) nonce, which must be unique for each message.
    let nonce = b"unique nonce";
    let plaintext = b"This is a top secret message.";

    println!("[ALICE] Encrypting message: '{}'", std::str::from_utf8(plaintext).unwrap());
    let encrypted_message = alice_channel.encrypt(plaintext, nonce).expect("Alice failed to encrypt");

    println!("[BOB]   Decrypting message...");
    let decrypted_message = bob_channel.decrypt(&encrypted_message, nonce).expect("Bob failed to decrypt");

    assert_eq!(plaintext, decrypted_message.as_slice());
    println!("SUCCESS: Bob decrypted the message: '{}'", std::str::from_utf8(&decrypted_message).unwrap());
}