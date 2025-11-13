#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/normalization.h"

namespace py = pybind11;

PYBIND11_MODULE(normalization, m) {
    m.doc() = "C++ vector normalization module for prompt optimizer";
    
    m.def("normalize_vector", &normalize_vector,
          "Normalize a single embedding vector using L2 normalization.\n\n"
          "Args:\n"
          "    embedding: List of floats representing the embedding vector\n\n"
          "Returns:\n"
          "    Normalized vector as a list of floats\n\n"
          "Example:\n"
          "    >>> import normalization\n"
          "    >>> vec = [3.0, 4.0, 0.0]\n"
          "    >>> normalized = normalization.normalize_vector(vec)\n"
          "    >>> # Result: [0.6, 0.8, 0.0]",
          py::arg("embedding"));
    
    m.def("normalize_vectors_batch", &normalize_vectors_batch,
          "Normalize multiple embedding vectors in batch.\n\n"
          "Args:\n"
          "    embeddings: List of lists of floats (list of embedding vectors)\n\n"
          "Returns:\n"
          "    List of normalized vectors\n\n"
          "Example:\n"
          "    >>> import normalization\n"
          "    >>> vecs = [[3.0, 4.0], [1.0, 1.0]]\n"
          "    >>> normalized = normalization.normalize_vectors_batch(vecs)",
          py::arg("embeddings"));
}

