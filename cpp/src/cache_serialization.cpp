#include "../include/cache_serialization.h"
#include <cstring>
#include <stdexcept>

std::vector<uint8_t> serialize_embedding(const std::vector<float>& embedding) {
    if (embedding.empty()) {
        std::vector<uint8_t> result(4, 0);
        return result;
    }
    
    size_t dimension = embedding.size();
    size_t total_size = 4 + (dimension * 4);
    std::vector<uint8_t> result(total_size);
    
    uint32_t dim = static_cast<uint32_t>(dimension);
    std::memcpy(result.data(), &dim, 4);
    std::memcpy(result.data() + 4, embedding.data(), dimension * 4);
    
    return result;
}

std::vector<float> deserialize_embedding(const std::vector<uint8_t>& data) {
    if (data.size() < 4) {
        throw std::runtime_error("Invalid binary data: too short (need at least 4 bytes for dimension)");
    }
    
    uint32_t dimension;
    std::memcpy(&dimension, data.data(), 4);
    
    if (dimension > 100000) {
        throw std::runtime_error("Invalid binary data: dimension too large (possible corruption)");
    }
    
    size_t expected_size = 4 + (dimension * 4);
    
    if (data.size() != expected_size) {
        throw std::runtime_error(
            "Invalid binary data: size mismatch (expected " + 
            std::to_string(expected_size) + " bytes, got " + 
            std::to_string(data.size()) + " bytes)"
        );
    }
    
    if (dimension == 0) {
        return std::vector<float>();
    }
    
    std::vector<float> result(dimension);
    std::memcpy(result.data(), data.data() + 4, dimension * 4);
    
    return result;
}

