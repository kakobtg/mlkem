# mlkem — ML-KEM-768 (FIPS 203 / CRYSTALS-Kyber) in Rust

**DISCLAIMER: This project is strictly for educational and learning purposes. It has not been audited by a third-party security firm and should NOT be used in production environments or to protect sensitive data. Cryptography is notoriously difficult to implement securely, especially regarding side-channel and timing attacks.**

## Table of contents

- [What is ML-KEM?](#what-is-ml-kem)
- [How ML-KEM-768 actually works](#how-ml-kem-768-actually-works)
- [Repository layout](#repository-layout)
- [Using this crate](#using-this-crate)
- [How to run everything](#how-to-run-everything)
  - [Run the examples](#run-the-examples)
  - [Run the tests](#run-the-tests)
  - [Run the official NIST KATs](#run-the-official-nist-kats)
- [Architecture & implementation notes](#architecture--implementation-notes)
- [Future improvements & security considerations](#future-improvements--security-considerations)

## What is ML-KEM?

**ML-KEM** (Module-Lattice-based Key-Encapsulation Mechanism) is the algorithm standardized by NIST in **FIPS 203**, based on the **CRYSTALS-Kyber** submission to NIST's post-quantum cryptography competition. It is a *post-quantum* algorithm: unlike RSA or elliptic-curve Diffie-Hellman, its security does not rely on integer factorization or the discrete-log problem — both of which a sufficiently large quantum computer running Shor's algorithm could break. Instead, ML-KEM's security rests on the hardness of the **Module Learning With Errors (Module-LWE)** problem over structured lattices, which is currently believed to resist quantum attack.

A **KEM** (Key-Encapsulation Mechanism) is the asymmetric-crypto building block used to agree on a shared secret between two parties, which is what public-key encryption is used for in almost every real protocol (TLS, SSH, etc.) — you don't encrypt the actual traffic with RSA/ECC/ML-KEM directly, you use it once to agree on a symmetric key, then switch to a fast symmetric cipher (e.g. AES-GCM) for the actual data. A KEM has exactly three operations:

| Operation | Who runs it | Input | Output |
|---|---|---|---|
| `KeyGen()` | Receiver | randomness | `(ek, dk)` — an encapsulation (public) key and a decapsulation (secret) key |
| `Encaps(ek)` | Sender | the receiver's public key | `(ct, ss)` — a ciphertext to send, and a shared secret only the sender currently knows |
| `Decaps(dk, ct)` | Receiver | its own secret key + the received ciphertext | `ss` — the same shared secret the sender derived |

This crate implements the **ML-KEM-768** parameter set, NIST's "security category 3" (roughly equivalent to AES-192).

## How ML-KEM-768 actually works

ML-KEM is built in two layers:

1. **K-PKE** — a public-key encryption scheme (not IND-CCA2 secure by itself) based on Module-LWE. It works over the polynomial ring `Z_q[X] / (X^256 + 1)` with `q = 3329`. A secret is a small vector of `k=3` such polynomials; the public key is `A·s + e` for a public random matrix `A` and small error `e` — recovering `s` from that is the Module-LWE problem.
2. **The Fujisaki-Okamoto (FO) transform** — wraps K-PKE to upgrade it to a full IND-CCA2-secure KEM. This is the part that makes ML-KEM resistant to *chosen-ciphertext* attacks: on `Decaps`, the implementation re-encrypts the recovered message and checks it reproduces the exact ciphertext it was given. If a ciphertext has been tampered with, instead of returning an error (which would leak information to an attacker probing the decryption oracle — a "padding oracle"), it deterministically derives a pseudo-random **implicit rejection** secret instead, indistinguishable from a real one to anyone without the secret key `z`. See [`src/kem.rs`](src/kem.rs) `decaps_internal_768` for exactly where this happens.

Concretely, the pipeline looks like this:

```
                        KeyGen                                     Encaps(ek)                          Decaps(dk, ct)
                     ┌────────────────┐                        ┌─────────────────────┐                ┌───────────────────────────┐
random seeds d,z ──▶│ sample A, s, e │                        │ sample r, e1, e2    │                │ recompute m' from ct + dk │
                     │ t = A·s + e    │──ek = (t, ρ)──────────▶│ u = Aᵀr + e1        │──ct = (u,v)───▶│ re-encrypt m' → ct'       │
                     │ ek = (t, ρ)    │                        │ v = tᵀr + e2 + m    │                │ ct' == ct ? real : reject │
                     │ dk = (s, ek,   │                        │ ct = compress(u, v) │                │ ss = K' or J(z ‖ c)       │
                     │      H(ek), z) │                        │ ss = K'             │                └───────────────────────────┘
                     └────────────────┘                        └─────────────────────┘
```

Note the shared secret is *not* hashed together with the ciphertext again — `Encaps` returns `K'` from `G(m ‖ H(ek))` directly, and `Decaps`'s implicit-rejection value is `J(z ‖ c)` over the raw ciphertext. (This trips people up because the original Kyber round-3 submission *did* mix in `H(c)` at this step — FIPS 203 simplified it away.)

Two implementation details make this efficient and compact:

- **The Number-Theoretic Transform (NTT)** turns the expensive `O(n²)` polynomial multiplication in the ring into an `O(n log n)` pointwise multiplication, the same trick as an FFT but over a finite field instead of complex numbers. This is what [`src/ntt.rs`](src/ntt.rs) implements.
- **Compression** (`Compress_d` / `Decompress_d` in FIPS 203) lossily rounds ciphertext coefficients down to `d` bits instead of the full 12-bit modulus, shrinking the ciphertext at the cost of a small, spec-bounded amount of noise the error-correction margin is designed to absorb. See [`src/encode.rs`](src/encode.rs).

The primary source this implementation follows is [FIPS 203](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf) itself; the NTT implementation notes in `ntt.rs` follow *"NEON NTT: Faster Dilithium, Kyber, and Saber on Cortex-A72 and Apple M1"* (Becker et al.).

## Repository layout

```
mlkem/
├── src/
│   ├── lib.rs        # Public API surface: keygen/encaps/decaps + the internal:: module for KATs
│   ├── params.rs      # MlKem768: every fixed constant from the spec (k, q, η, du/dv, byte sizes)
│   ├── kem.rs         # The ML-KEM layer: KeyGen/Encaps/Decaps + the FO transform (implicit rejection)
│   ├── pke.rs          # The underlying K-PKE layer: KeyGen/Encrypt/Decrypt (Module-LWE)
│   ├── ntt.rs          # Forward/inverse NTT and pointwise multiplication in the NTT domain
│   ├── poly.rs         # The `Poly` type (256 coefficients) and basic ring arithmetic (add/sub)
│   ├── reduce.rs       # Modular reduction helpers (plain mod, Barrett, Montgomery)
│   ├── sample.rs       # Turns hash output into polynomials: rejection sampling (SampleNTT) and
│   │                   # centered binomial distribution sampling (the "noise")
│   ├── encode.rs       # ByteEncode/ByteDecode (bit-packing) and Compress/Decompress
│   ├── hash.rs         # The hash/XOF functions the spec names H, G, J and PRF (SHA3-256/512, SHAKE128/256)
│   ├── ct.rs            # Constant-time equality/select helpers (via `subtle`), used by implicit rejection
│   ├── util.rs         # Tiny byte-splitting helper
│   ├── error.rs        # `MlKemError` (currently unused in practice — the reference algorithms don't
│   │                   # actually fail for well-formed fixed-size inputs)
│   └── smoke.rs        # A duplicate of tests/smoke.rs kept inside src/ (not wired into lib.rs)
├── examples/
│   ├── kem_roundtrip.rs         # Minimal keygen → encaps → decaps example
│   └── three_party_exchange.rs # Alice/Bob/Eve narrative demo, plus using the shared secret for AES-GCM
├── tests/
│   ├── smoke.rs        # Basic correctness tests: roundtrip, implicit rejection, key mismatch
│   ├── pipeline.rs     # One large end-to-end integration test — see below
│   └── nist_kats.rs    # Runs the official NIST ACVP test vectors (needs `test-utils` feature)
└── test_vectors/
    └── ML-KEM-768.json # Official NIST ACVP test vectors consumed by tests/nist_kats.rs
```

A few files are worth calling out specifically:

- **[`src/lib.rs`](src/lib.rs)** is the whole public API: `keygen`, `encaps`, `decaps`, and the fixed-size type aliases `Ek`/`Dk`/`Ct`/`Ss` (plain byte arrays — no heap allocation, so this works in `no_std`). It also exposes a `mod internal` (only compiled with the `test-utils` feature) with *deterministic* versions of the same three functions that take explicit random seeds instead of an RNG — this is what lets `tests/nist_kats.rs` reproduce the official test vectors exactly.
- **[`src/kem.rs`](src/kem.rs)** vs **[`src/pke.rs`](src/pke.rs)**: `pke.rs` is the raw Module-LWE encryption scheme (Algorithm 13/14/15 in FIPS 203); `kem.rs` wraps it with the FO transform to get CCA2 security (Algorithm 16/17/18). If you're trying to understand *why* ML-KEM is secure against an active attacker, `kem.rs` is where that logic lives; if you're trying to understand the underlying lattice math, `pke.rs` is where that lives.
- **[`src/ntt.rs`](src/ntt.rs)** is by far the largest file (~1,100 lines) because it also carries the AArch64 NEON SIMD-oriented implementation notes and twiddle-factor tables described in the header doc comment — worth reading if you care about how this is made fast, not just correct.
- **[`tests/pipeline.rs`](tests/pipeline.rs)** is a single comprehensive test that drives the library the way a real caller would: keygen → serialize the public key across a simulated "wire" → encapsulate → serialize the ciphertext back → decapsulate → use the resulting shared secret as a real AES-256-GCM key, plus the adversarial cases (an eavesdropper with the wrong key, a bit-flipped ciphertext, wrong keypairs) and repeated independent sessions to check secrets never repeat. It's meant to be the one file to read if you want to see the whole system exercised in one place.
- **[`tests/nist_kats.rs`](tests/nist_kats.rs)** replays the official NIST ACVP vectors for ML-KEM-768 — `test_vectors/ML-KEM-768.json` — against the deterministic `internal::` API: 25 keyGen vectors, 25 encapsulation vectors, and 10 decapsulation vectors (including "modified ciphertext" cases that specifically exercise implicit rejection). The JSON was pulled straight from NIST's own [`usnistgov/ACVP-Server`](https://github.com/usnistgov/ACVP-Server) repository and filtered down to just the ML-KEM-768 groups; the file itself records the exact source path.

## Using this crate

```toml
[dependencies]
mlkem = { git = "https://github.com/kakobtg/mlkem" }
rand = "0.8"
```

```rust
use mlkem::{keygen, encaps, decaps};
use rand::rngs::OsRng;

fn main() {
    let mut rng = OsRng;

    // 1. Generate a public/private keypair
    let keypair = keygen(&mut rng);

    // 2. Encapsulate a shared secret against the public encapsulation key
    let (ciphertext, shared_secret_sender) = encaps(&mut rng, &keypair.ek)
        .expect("Encapsulation failed");

    // 3. Decapsulate the ciphertext using the private decapsulation key
    let shared_secret_receiver = decaps(&keypair.dk, &ciphertext)
        .expect("Decapsulation failed");

    assert_eq!(shared_secret_sender, shared_secret_receiver);
    println!("Successfully established post-quantum shared secret!");
}
```

## How to run everything

All commands below are run from this repository's root.

### Run the examples

```bash
cargo run --example kem_roundtrip
cargo run --example three_party_exchange
```

### Run the tests

```bash
# Unit tests, component smoke tests, and the end-to-end pipeline test:
cargo test
```

### Run the official NIST KATs

The vector file `test_vectors/ML-KEM-768.json` is already present in this repo (official NIST ACVP data for ML-KEM-768, keyGen + encapsulation + decapsulation). Running the KAT harness requires the `test-utils` feature, since it needs the deterministic `internal::` API (real callers should never construct keys/messages from fixed seeds — that API only exists for exactly this purpose):

```bash
cargo test --features test-utils --test nist_kats
```

This replays 25 keyGen, 25 encapsulation, and 10 decapsulation vectors and asserts this implementation reproduces every one exactly, byte for byte.

## Architecture & implementation notes

* **Modular arithmetic:** all mathematical operations are strictly bounded within the ring `Z_q[X]/(X^256 + 1)` where `q = 3329` (`src/reduce.rs`, `src/poly.rs`).
* **Montgomery domain:** the NEON-SIMD-oriented NTT code performs arithmetic in the Montgomery domain (`R = 2^16`) to avoid expensive modular reduction instructions on hot paths (`src/ntt.rs`).
* **Implicit rejection:** as described above, `decaps` never returns a decryption-failure error for a tampered ciphertext — it returns a deterministic pseudo-random secret instead, which is what makes the FO transform resistant to chosen-ciphertext/padding-oracle attacks (`src/kem.rs`, `src/ct.rs`).
* **`no_std`-capable:** the crate is `#![no_std]` unless the `std` feature is enabled (it's on by default), and every public type is a fixed-size byte array — no heap allocation on the core keygen/encaps/decaps path.

## Future improvements & security considerations

As an educational project, there are several areas that require improvement before this crate could be considered cryptographically secure for real-world usage:

1. **Strict constant-time execution:**
   While the mathematical NTT structures are naturally constant-time, certain operations (especially the implicit rejection comparison in `decaps_internal` and polynomial compression/decompression bounds) must be rigorously vetted and refactored using a crate like `subtle` to ensure they do not leak timing information.
2. **Secure memory zeroization:**
   Cryptographic keys (especially `dk` and intermediate shared secrets) currently reside in standard memory. A production implementation must use the `zeroize` crate more thoroughly to securely wipe memory buffers the moment they go out of scope, preventing cold-boot and memory-dump attacks.
3. **Side-channel attack (SCA) resistance:**
   Advanced side-channel protections, such as coefficient masking and execution jitter, are not implemented. An attacker with physical access to the device or power-monitoring capabilities could potentially extract secret keys.
4. **x86_64 optimizations (AVX2 / AVX-512):**
   Currently, SIMD optimizations are strictly written for AArch64 (NEON). Adding AVX2 and AVX-512 intrinsics would dramatically improve performance on standard desktop and server architectures.
5. **Formal verification:**
   Using tools like `hax` or `frama-c` to formally verify the mathematical equivalence of the optimized SIMD operations against the reference specification.
6. **FIPS 203 encapsulation/decapsulation key checks:**
   The spec requires implementations to reject non-canonical encapsulation keys (the "modulus check": `ByteEncode12(ByteDecode12(ek)) == ek`) and to verify the hash embedded in a decapsulation key against a freshly computed one (the "hash check"). Neither is implemented here yet. The ACVP data pulled into `test_vectors/ML-KEM-768.json` already includes `encapsulationKeyCheck`/`decapsulationKeyCheck` groups (10 vectors each) that specifically exercise this — they're not wired into `tests/nist_kats.rs` yet because there's nothing in the implementation for them to validate.
