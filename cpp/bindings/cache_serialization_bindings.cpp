#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/cache_serialization.h"

namespace py = pybind11;

PYBIND11_MODULE(cache_serialization, m) {
    m.doc() = "C++ binary serialization module for embedding vectors";
    
    m.def("serialize_embedding", &serialize_embedding,
          "Serialize an embedding vector to binary format.\n\n"
          "Binary format: [dimension (4 bytes)][float values (4 bytes each)]\n\n"
          "Args:\n"
          "    embedding: List of floats representing the embedding vector\n\n"
          "Returns:\n"
          "    Binary data as bytes object\n\n"
          "Example:\n"
          "    >>> import cache_serialization\n"
          "    >>> vec = [1.0, 2.0, 3.0]\n"
          "    >>> binary = cache_serialization.serialize_embedding(vec)\n"
          "    >>> len(binary)  # 4 (dimension) + 3*4 (floats) = 16 bytes",
          py::arg("embedding"));
    
    m.def("deserialize_embedding", [](py::bytes data) {
        std::string buffer = data.cast<std::string>();
        std::vector<uint8_t> byte_vector(buffer.begin(), buffer.end());
        return deserialize_embedding(byte_vector);
    }, "Deserialize binary data to embedding vector.\n\n"
          "Args:\n"
          "    data: Binary data (bytes object) from serialize_embedding()\n\n"
          "Returns:\n"
          "    Embedding vector as list of floats\n\n"
          "Raises:\n"
          "    RuntimeError: If data is invalid or corrupted\n\n"
          "Example:\n"
          "    >>> import cache_serialization\n"
          "    >>> binary = b'...'  # from serialize_embedding\n"
          "    >>> vec = cache_serialization.deserialize_embedding(binary)",
          py::arg("data"));
}

