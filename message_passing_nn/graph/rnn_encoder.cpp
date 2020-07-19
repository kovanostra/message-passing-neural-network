#include <torch/extension.h>
#include "../utils/messages.h"
#include "../utils/derivatives.h"
#include "../utils/array_operations.h"
#include <iostream>

std::vector<torch::Tensor> forward_cpp(
    const int& time_steps,
    const int& number_of_nodes,
    const int& number_of_node_features,
    const int& fully_connected_layer_output_size,
    const int& batch_size,
    const torch::Tensor& node_features,
    const torch::Tensor& all_neighbors,
    const torch::Tensor& w_graph_node_features,
    const torch::Tensor& w_graph_neighbor_messages,
    const torch::Tensor& u_graph_node_features,
    const torch::Tensor& u_graph_neighbor_messages,
    const torch::Tensor& linear_weight,
    const torch::Tensor& linear_bias) {
      
    auto outputs = torch::zeros({batch_size, fully_connected_layer_output_size});
    auto linear_outputs = torch::zeros({batch_size, fully_connected_layer_output_size});
    auto messages = torch::zeros({batch_size, number_of_nodes, number_of_nodes, number_of_node_features});
    auto messages_previous_step = torch::zeros({batch_size, number_of_nodes, number_of_nodes, number_of_node_features});
    auto node_encoding_messages = torch::zeros({batch_size, number_of_nodes, number_of_node_features});
    auto encodings = torch::zeros({batch_size, number_of_nodes*number_of_node_features});
    
      
    for (int batch = 0; batch<batch_size; batch++) {
      auto messages_vector = compose_messages(time_steps,
                                        number_of_nodes,
                                        number_of_node_features,
                                        w_graph_node_features,
                                        w_graph_neighbor_messages,
                                        node_features[batch],
                                        all_neighbors[batch],
                                        messages[batch]);
      messages[batch] += messages_vector[0];
      messages_previous_step[batch] += messages_vector[1];
      encodings[batch] += encode_messages(number_of_nodes,
                                        node_encoding_messages[batch],
                                        u_graph_node_features,
                                        u_graph_neighbor_messages,
                                        node_features[batch],
                                        all_neighbors[batch],
                                        messages[batch]).view({-1});
      linear_outputs[batch] += linear_bias.add_(torch::matmul(linear_weight, encodings[batch]));
      outputs[batch] += linear_outputs[batch].sigmoid_();
    }
    return {outputs, linear_outputs, encodings, messages, messages_previous_step};
  }

std::vector<torch::Tensor> backward_cpp(
  const torch::Tensor& grad_output,
  const torch::Tensor& outputs,
  const torch::Tensor& linear_outputs,
  const torch::Tensor& encodings,
  const torch::Tensor& messages_summed,
  const torch::Tensor& messages_previous_step_summed,
  const torch::Tensor& messages,
  const torch::Tensor& node_features,
  const torch::Tensor& batch_size,
  const torch::Tensor& number_of_nodes,
  const torch::Tensor& number_of_node_features,
  const torch::Tensor& u_graph_neighbor_messages_summed,
  const torch::Tensor& linear_weight,
  const torch::Tensor& linear_bias) {
  
  auto delta_1 = grad_output*d_sigmoid(linear_outputs);
  auto d_linear_bias = delta_1;
  auto d_linear_weight = torch::matmul(delta_1.transpose(0, 1), encodings);
  
  auto delta_2 = torch::matmul(delta_1, linear_weight).reshape({batch_size.item<int>(), number_of_nodes.item<int>(), number_of_node_features.item<int>()})*(d_relu_2d(encodings).reshape({batch_size.item<int>(), number_of_nodes.item<int>(), number_of_node_features.item<int>()}));
  auto d_u_graph_node_features = torch::matmul(delta_2, node_features.transpose(1, 2));
  auto d_u_graph_neighbor_messages = torch::matmul(delta_2.transpose(1, 2), messages_summed);

  auto delta_3 = torch::matmul(delta_2.transpose(1, 2), torch::matmul(u_graph_neighbor_messages_summed, d_relu_4d(messages).transpose(2, 3)));
  auto d_w_graph_node_features = torch::matmul(delta_3, node_features);
  auto d_w_graph_neighbor_messages = torch::matmul(delta_3, messages_previous_step_summed);


  return {d_w_graph_node_features, 
          d_w_graph_neighbor_messages, 
          d_u_graph_node_features, 
          d_u_graph_neighbor_messages,
          d_linear_weight,
          d_linear_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_cpp, "RNN encoder forward pass (CPU)");
  m.def("backward", &backward_cpp, "RNN encoder backward pass (CPU)");
  m.def("compose_messages", &compose_messages, "RNN compose messages (CPU)");
  m.def("encode_messages", &encode_messages, "RNN encode messages (CPU)");
}