#[derive(Debug, Clone)]
pub enum MlKemError {
    InvalidEncapsulationKey,
    InvalidDecapsulationKey,
    InvalidCiphertext,
}

