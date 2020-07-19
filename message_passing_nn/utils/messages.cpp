#include <torch/extension.h>
#include "../utils/messages.h"
#include "../utils/array_operations.h"

torch::Tensor remove_element_from_tensor(const torch::Tensor& tensor,
                                         int& element_index){
    torch::Tensor new_tensor = torch::empty({tensor.sizes()[0] - 1});
    if (element_index == 0){
      new_tensor.copy_(tensor.index({torch::indexing::Slice(1, tensor.sizes()[0])}));
    } else if (element_index == tensor.sizes()[0]) {
      new_tensor.copy_(tensor.index({torch::indexing::Slice(0, tensor.sizes()[0] - 1)}));
    } else {
      new_tensor.copy_(torch::cat({tensor.index({torch::indexing::Slice(0, element_index)}), 
                                   tensor.index({torch::indexing::Slice(element_index + 1, tensor.sizes()[0])})}));
    } 
  return new_tensor;
}

torch::Tensor compute_messages_from_neighbors(const torch::Tensor& all_neighbors_of_node,
                                              const int& node_id,
                                              const int& end_node_id,
                                              int& end_node_index,
                                              const torch::Tensor& w_graph_neighbor_messages,
                                              const torch::Tensor& messages_previous_step){
  torch::Tensor messages_from_the_other_neighbors = torch::zeros_like({messages_previous_step[0][0]});
  auto other_neighbors = remove_element_from_tensor(all_neighbors_of_node, end_node_index);
  for (int neighbor_index = 0; neighbor_index<other_neighbors.sizes()[0]; neighbor_index++) {
    auto neighbor = other_neighbors[neighbor_index].item<int64_t>();
    if (neighbor >= 0) {
      messages_from_the_other_neighbors += torch::matmul(w_graph_neighbor_messages, messages_previous_step[neighbor][node_id]);
    }
  }
  return messages_from_the_other_neighbors;
}

torch::Tensor get_messages_to_all_end_nodes(const int& node_id,
                                            const torch::Tensor& w_graph_neighbor_messages,
                                            const torch::Tensor& w_graph_node_features,
                                            const torch::Tensor& all_neighbors_of_node,
                                            const torch::Tensor& features_of_specific_node,
                                            const torch::Tensor& messages_previous_step,
                                            torch::Tensor& new_messages_of_node) {
  for (int end_node_index = 0; end_node_index<all_neighbors_of_node.sizes()[0]; end_node_index++){
      auto end_node_id = all_neighbors_of_node[end_node_index].item<int64_t>();
      if (end_node_id >= 0) {
        auto messages_from_the_other_neighbors = compute_messages_from_neighbors(all_neighbors_of_node,
                                                                                 node_id,
                                                                                 end_node_id,
                                                                                 end_node_index,
                                                                                 w_graph_neighbor_messages,
                                                                                 messages_previous_step);
        new_messages_of_node[end_node_id] += torch::matmul(w_graph_node_features, features_of_specific_node).add_(messages_from_the_other_neighbors);
      }
    }
  return new_messages_of_node;
}

std::vector<torch::Tensor> compose_messages(
    const int& time_steps,
    const int& number_of_nodes,
    const int& number_of_node_features,
    const torch::Tensor& w_graph_node_features,
    const torch::Tensor& w_graph_neighbor_messages,
    const torch::Tensor& node_features,
    const torch::Tensor& all_neighbors,
    const torch::Tensor& messages_init) {

  auto new_messages = torch::zeros_like({messages_init});
  auto messages_previous_step = torch::zeros_like({messages_init});
  auto new_messages_of_node = torch::zeros({messages_previous_step.sizes()[1], messages_previous_step.sizes()[2]});

  for (int time_step = 0; time_step<time_steps; time_step++) {
    messages_previous_step.copy_(new_messages);
    new_messages.zero_();
    for (int node_id = 0; node_id<number_of_nodes; node_id++) {
      new_messages[node_id] += get_messages_to_all_end_nodes(node_id,
                                                             w_graph_neighbor_messages,
                                                             w_graph_node_features,
                                                             all_neighbors[node_id],
                                                             node_features[node_id],
                                                             messages_previous_step,
                                                             new_messages_of_node);
      new_messages_of_node.zero_();
    }
  }
  return {new_messages, messages_previous_step};
}

torch::Tensor encode_messages(
    const int& number_of_nodes,
    const torch::Tensor& node_encoding_messages,
    const torch::Tensor& u_graph_node_features,
    const torch::Tensor& u_graph_neighbor_messages,
    const torch::Tensor& node_features,
    const torch::Tensor& all_neighbors,
    const torch::Tensor& messages) {

    for (int node_id = 0; node_id<number_of_nodes; node_id++) {
      for (int end_node_index = 0; end_node_index<all_neighbors.sizes()[0]; end_node_index++){
        auto end_node_id = all_neighbors[node_id][end_node_index].item<int64_t>();
        if (end_node_id >= 0) {
          node_encoding_messages[node_id] += torch::matmul(u_graph_neighbor_messages, messages[end_node_id][node_id].relu_());
        }
      }
    }
    return torch::matmul(u_graph_node_features, node_features).add_(node_encoding_messages).relu_();
  }
