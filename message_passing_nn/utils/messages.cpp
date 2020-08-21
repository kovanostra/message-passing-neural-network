#include <torch/extension.h>
#include "../utils/messages.h"

std::vector<at::Tensor> compose_messages(
    const int& time_steps,
    const int& number_of_nodes,
    const int& number_of_node_features,
    const at::Tensor& w_graph_node_features,
    const at::Tensor& w_graph_neighbor_messages,
    const at::Tensor& base_messages,
    const at::Tensor& all_neighbors,
    const at::Tensor& messages_init) {

  auto new_messages = at::zeros_like({messages_init});
  auto previous_messages = at::zeros_like({messages_init});
  auto new_messages_of_node = at::zeros({previous_messages.sizes()[1], previous_messages.sizes()[2]});
  for (int time_step = 0; time_step<time_steps; time_step++) {
    auto base_neighbor_messages = at::matmul(w_graph_neighbor_messages, at::relu(new_messages));
    std::swap(previous_messages, new_messages);
    for (int node_id = 0; node_id < all_neighbors.size(0); node_id++) {
      for (int end_node_index = 0; end_node_index < all_neighbors.size(1); end_node_index++){
        auto end_node_id = all_neighbors[node_id][end_node_index].item<int>();
        if (end_node_id >= 0) {
          new_messages[node_id][end_node_id] += base_messages[node_id];
          for (int neighbor_index = 0; neighbor_index < all_neighbors.size(1); neighbor_index++) {
            auto neighbor = all_neighbors[node_id][neighbor_index].item<int>();
            if (neighbor >= 0 && neighbor_index!=end_node_index) {
              new_messages[node_id][end_node_id] += base_neighbor_messages[neighbor][node_id];
            }
          }
        }
      }
    }
  }
  return {new_messages, previous_messages};
}

at::Tensor encode_messages(
    const int& number_of_nodes,
    const at::Tensor& node_encoding_messages,
    const at::Tensor& u_graph_node_features,
    const at::Tensor& u_graph_neighbor_messages,
    const at::Tensor& node_features,
    const at::Tensor& all_neighbors,
    const at::Tensor& messages) {

    for (int node_id = 0; node_id<number_of_nodes; node_id++) {
      for (int end_node_index = 0; end_node_index<all_neighbors.sizes()[1]; end_node_index++){
        auto end_node_id = all_neighbors[node_id][end_node_index].item<int64_t>();
        if (end_node_id >= 0) {
          node_encoding_messages[node_id] += at::matmul(u_graph_neighbor_messages, at::relu(messages[end_node_id][node_id]));
        }
      }
    }
    return at::relu(at::add(at::matmul(u_graph_node_features, node_features), node_encoding_messages));
  }
