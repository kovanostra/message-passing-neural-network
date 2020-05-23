#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <iostream>

at::Tensor* compose_messages(torch::Tensor node_features,
                                torch::Tensor adjacency_matrix,
                                at::Tensor* messages) {
    auto new_messages = new torch::Tensor[adjacency_matrix.size(0), adjacency_matrix.size(1), node_features.size(1)];
    return new_messages;

}

torch::Tensor* rnn_encoder_forward(
    torch::Tensor node_features,
    torch::Tensor adjacency_matrix,
    int16_t batch_size) {

  auto outputs = new torch::Tensor[batch_size];
  auto messages = new torch::Tensor[adjacency_matrix.size(0), adjacency_matrix.size(1), node_features.size(1)];

  for(int batch=0; batch<batch_size ; batch++)
  {
     outputs[batch] = torch::sigmoid(torch::add(adjacency_matrix, adjacency_matrix));
     messages = compose_messages(node_features, adjacency_matrix, messages);
  }

  return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rnn_encoder_forward, "RNN encoder forward (CPU)");
}