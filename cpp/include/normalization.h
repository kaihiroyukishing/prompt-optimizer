#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include <vector>

/**
 * Normalize a single embedding vector using L2 normalization.
 * 
 * Normalizes the vector to unit length. If the vector is zero (all zeros),
 * it remains zero (no division by zero error).
 * 
 * @param embedding Vector to normalize
 * @return Normalized vector (new vector, input not modified)
 */
std::vector<float> normalize_vector(const std::vector<float>& embedding);

/**
 * Normalize multiple embedding vectors in batch.
 * 
 * Normalizes each vector in the batch to unit length. This is more efficient
 * than calling normalize_vector() multiple times due to better cache locality.
 * 
 * @param embeddings Vector of vectors to normalize
 * @return Vector of normalized vectors (new vectors, input not modified)
 */
std::vector<std::vector<float>> normalize_vectors_batch(
    const std::vector<std::vector<float>>& embeddings
);

#endif // NORMALIZATION_H

