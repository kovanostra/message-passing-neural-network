#include <torch/extension.h>
#include "../utils/messages.h"
#include "../utils/array_operations.h"


std::vector<torch::Tensor> compose_messages(
    const int& time_steps,
    const int& number_of_nodes,
    const int& number_of_node_features,
    const torch::Tensor& w_graph_node_features,
    const torch::Tensor& w_graph_neighbor_messages,
    const torch::Tensor& node_features,
    const torch::Tensor& adjacency_matrix,
    const torch::Tensor& messages_init) {

  at::Tensor new_messages = torch::zeros_like({messages_init});
  at::Tensor messages_previous_step;
  at::Tensor messages_from_the_other_neighbors;
  std::vector<int> all_neighbors;
  int number_of_neighbors;
  int end_node_index;
  std::vector<int> other_neighbors;

  for (int time_step = 0; time_step<time_steps; time_step++) {
    messages_previous_step = new_messages;
    new_messages = torch::zeros_like({messages_init});

    for (int node_id = 0; node_id<number_of_nodes; node_id++) {
      all_neighbors = find_nonzero_elements(adjacency_matrix[node_id]);
      number_of_neighbors = static_cast<int>(all_neighbors.size());
      
      for (int end_node_id: all_neighbors){
        messages_from_the_other_neighbors = torch::zeros_like({messages_init[0][0]});

        if (number_of_neighbors > 1) {
          end_node_index = find_index_by_value(all_neighbors, end_node_id);
          other_neighbors = remove_element_by_index_from_vector(all_neighbors, end_node_index);      
          for (int neighbor: other_neighbors) {
              messages_from_the_other_neighbors += torch::matmul(w_graph_neighbor_messages, 
                                                                 torch::relu(messages_previous_step[neighbor][node_id]));
          }
        }
        new_messages[node_id][end_node_id] = torch::add(torch::matmul(w_graph_node_features, node_features[node_id]), 
                                                                    messages_from_the_other_neighbors);
      }
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
    const torch::Tensor& adjacency_matrix,
    const torch::Tensor& messages) {

    std::vector<int> all_neighbors;
      
    for (int node_id = 0; node_id<number_of_nodes; node_id++) {
      all_neighbors = find_nonzero_elements(adjacency_matrix[node_id]);

      for (int end_node_id: all_neighbors){
        node_encoding_messages[node_id] += torch::matmul(u_graph_neighbor_messages, messages[end_node_id][node_id]);
      }
    }
    return torch::relu(torch::add(torch::matmul(u_graph_node_features, node_features), node_encoding_messages));
  }
