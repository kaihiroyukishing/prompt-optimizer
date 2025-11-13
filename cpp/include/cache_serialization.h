#ifndef CACHE_SERIALIZATION_H
#define CACHE_SERIALIZATION_H

#include <string>
#include <vector>
#include <cstdint>

/**
 * Serialize a single embedding vector to binary format.
 * 
 * Binary format: [dimension (4 bytes uint32)][float values (4 bytes each)]
 * 
 * @param embedding Vector to serialize
 * @return Binary data as vector of bytes
 */
std::vector<uint8_t> serialize_embedding(const std::vector<float>& embedding);

/**
 * Deserialize binary data to embedding vector.
 * 
 * @param data Binary data to deserialize
 * @return Deserialized embedding vector
 * @throws std::runtime_error if data is invalid or corrupted
 */
std::vector<float> deserialize_embedding(const std::vector<uint8_t>& data);

#endif // CACHE_SERIALIZATION_H

