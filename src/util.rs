pub fn split_at_32<const N: usize>(x: &[u8; N], offset: usize) -> ([u8; 32], &[u8]) {
    let mut left = [0u8; 32];
    left.copy_from_slice(&x[offset..offset + 32]);
    (left, &x[offset + 32..])
}
