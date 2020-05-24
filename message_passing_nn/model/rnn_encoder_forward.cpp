#include <torch/extension.h>
#include <iostream>

at::Tensor messages_from_the_other_neighbors(
    const at::Tensor& w_graph_neighbor_messages,
    const at::Tensor& messages,
    const int& node_id,
    const at::Tensor& all_neighbors,
    const int& end_node_index) {

  at::Tensor messages_from_the_other_neighbors = at::zeros_like({messages[0][0]});

  at::Tensor other_neighbors = torch::cat({all_neighbors.slice(0, 0, end_node_index), 
                                           all_neighbors.slice(0, end_node_index + 1, all_neighbors.sizes()[0])});
  for (int z = 0; z < other_neighbors.sizes()[0]; ++z) {
      auto neighbor = other_neighbors[z].item<int>();
      messages_from_the_other_neighbors += torch::matmul(w_graph_neighbor_messages, messages[neighbor][node_id]);
  }

  return messages_from_the_other_neighbors;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("messages_from_the_other_neighbors", &messages_from_the_other_neighbors, "RNN encoder forward (CPU)");
}