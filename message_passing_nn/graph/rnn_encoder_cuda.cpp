#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> forward_cuda_cpp(
    const at::Tensor& time_steps,
    const at::Tensor& number_of_nodes,
    const at::Tensor& number_of_node_features,
    const at::Tensor& fully_connected_layer_output_size,
    const at::Tensor& batch_size,
    const at::Tensor& node_features,
    const at::Tensor& all_neighbors,
    const at::Tensor& w_graph_node_features,
    const at::Tensor& w_graph_neighbor_messages,
    const at::Tensor& u_graph_node_features,
    const at::Tensor& u_graph_neighbor_messages,
    const at::Tensor& linear_weight,
    const at::Tensor& linear_bias);

std::vector<torch::Tensor> backward_cuda_cpp(
    const at::Tensor& grad_output,
    const at::Tensor& outputs,
    const at::Tensor& linear_outputs,
    const at::Tensor& encodings,
    const at::Tensor& messages_summed,
    const at::Tensor& messages_previous_step_summed,
    const at::Tensor& messages,
    const at::Tensor& node_features,
    const at::Tensor& batch_size,
    const at::Tensor& number_of_nodes,
    const at::Tensor& number_of_node_features,
    const at::Tensor& u_graph_neighbor_messages_summed,
    const at::Tensor& linear_weight,
    const at::Tensor& linear_bias);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> forward_cpp(
    const at::Tensor& time_steps,
    const at::Tensor& number_of_nodes,
    const at::Tensor& number_of_node_features,
    const at::Tensor& fully_connected_layer_output_size,
    const at::Tensor& batch_size,
    const at::Tensor& node_features,
    const at::Tensor& all_neighbors,
    const at::Tensor& w_graph_node_features,
    const at::Tensor& w_graph_neighbor_messages,
    const at::Tensor& u_graph_node_features,
    const at::Tensor& u_graph_neighbor_messages,
    const at::Tensor& linear_weight,
    const at::Tensor& linear_bias) {
    CHECK_INPUT(time_steps);
    CHECK_INPUT(number_of_nodes);
    CHECK_INPUT(number_of_node_features);
    CHECK_INPUT(fully_connected_layer_output_size);
    CHECK_INPUT(batch_size);
    CHECK_INPUT(node_features);
    CHECK_INPUT(all_neighbors);
    CHECK_INPUT(w_graph_node_features);
    CHECK_INPUT(w_graph_neighbor_messages);
    CHECK_INPUT(u_graph_node_features);
    CHECK_INPUT(u_graph_neighbor_messages);
    CHECK_INPUT(linear_weight);
    CHECK_INPUT(linear_bias);

  return forward_cuda_cpp(time_steps,
    number_of_nodes,
    number_of_node_features,
    fully_connected_layer_output_size,
    batch_size,
    node_features,
    all_neighbors,
    w_graph_node_features,
    w_graph_neighbor_messages,
    u_graph_node_features,
    u_graph_neighbor_messages,
    linear_weight,
    linear_bias);
}

std::vector<torch::Tensor> backward_cpp(
    const at::Tensor& grad_output,
    const at::Tensor& outputs,
    const at::Tensor& linear_outputs,
    const at::Tensor& encodings,
    const at::Tensor& messages_summed,
    const at::Tensor& messages_previous_step_summed,
    const at::Tensor& messages,
    const at::Tensor& node_features,
    const at::Tensor& batch_size,
    const at::Tensor& number_of_nodes,
    const at::Tensor& number_of_node_features,
    const at::Tensor& u_graph_neighbor_messages_summed,
    const at::Tensor& linear_weight,
    const at::Tensor& linear_bias) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(outputs);
    CHECK_INPUT(linear_outputs);
    CHECK_INPUT(encodings);
    CHECK_INPUT(messages_summed);
    CHECK_INPUT(messages_previous_step_summed);
    CHECK_INPUT(messages);
    CHECK_INPUT(node_features);
    CHECK_INPUT(batch_size);
    CHECK_INPUT(number_of_nodes);
    CHECK_INPUT(number_of_node_features);
    CHECK_INPUT(u_graph_neighbor_messages_summed);
    CHECK_INPUT(linear_weight);
    CHECK_INPUT(linear_bias);

  return backward_cuda_cpp(
      grad_output,
      outputs,
      linear_outputs,
      encodings,
      messages_summed,
      messages_previous_step_summed,
      messages,
      node_features,
      batch_size,
      number_of_nodes,
      number_of_node_features,
      u_graph_neighbor_messages_summed,
      linear_weight,
      linear_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_cpp, "RNN encoder forward pass (CUDA)");
  m.def("backward", &backward_cpp, "RNN encoder backward pass (CUDA)");
}