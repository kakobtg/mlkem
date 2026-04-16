/// ML-KEM-768 parameters (hard-coded default)
pub struct MlKem768;

impl MlKem768 {
    pub const K: usize = 3;
    pub const N: usize = 256;
    pub const Q: i16 = 3329;

    // noise parameters
    pub const ETA1: usize = 2;
    pub const ETA2: usize = 2;

    // compression parameters
    pub const DU: usize = 10;
    pub const DV: usize = 4;

    // sizes
    pub const SS_BYTES: usize = 32;
    pub const SEED_BYTES: usize = 32;

    pub const EK_BYTES: usize = 1184;
    pub const DK_BYTES: usize = 2400;
    pub const CT_BYTES: usize = 1088;

    // derived sizes used by encodings
    pub const POLY_BYTES_12: usize = 384; // 256 * 12 bits = 3072 bits = 384 bytes
}
