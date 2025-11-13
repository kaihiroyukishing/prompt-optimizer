#include "../include/normalization.h"
#include <cmath>
#include <algorithm>

std::vector<float> normalize_vector(const std::vector<float>& embedding) {
    if (embedding.empty()) {
        return embedding;
    }
    
    float norm = 0.0f;
    for (float value : embedding) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    
    if (norm == 0.0f) {
        return embedding;
    }
    
    std::vector<float> result;
    result.reserve(embedding.size());
    
    for (float value : embedding) {
        result.push_back(value / norm);
    }
    
    return result;
}

std::vector<std::vector<float>> normalize_vectors_batch(
    const std::vector<std::vector<float>>& embeddings
) {
    std::vector<std::vector<float>> result;
    result.reserve(embeddings.size());
    
    for (const auto& embedding : embeddings) {
        result.push_back(normalize_vector(embedding));
    }
    
    return result;
}

