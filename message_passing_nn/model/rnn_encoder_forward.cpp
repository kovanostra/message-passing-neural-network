#include <torch/extension.h>
#include <iostream>

at::Tensor messages_from_the_other_neighbors(
    const at::Tensor& w_graph_neighbor_messages,
    const at::Tensor& messages) {

  auto messages_from_the_other_neighbors = torch::matmul(w_graph_neighbor_messages, messages);
  for (auto const& i : data) {
      std::cout << i.name;
  }

  return messages_from_the_other_neighbors;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("messages_from_the_other_neighbors", &messages_from_the_other_neighbors, "RNN encoder forward (CPU)");
}