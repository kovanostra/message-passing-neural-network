#ifndef MESSAGES_H
#define MESSAGES_H

#include <torch/extension.h>

torch::Tensor compute_messages_from_neighbors(std::vector<int> all_neighbors,
                                              const int& node_id,
                                              const int& end_node_id,
                                              const torch::Tensor& w_graph_neighbor_messages,
                                              const torch::Tensor& messages_previous_step);

torch::Tensor get_messages_to_all_end_nodes(const int& node_id,
                                            const torch::Tensor& w_graph_neighbor_messages,
                                            const torch::Tensor& w_graph_node_features,
                                            const torch::Tensor& adjacency_vector_of_specific_node,
                                            const torch::Tensor& features_of_specific_node,
                                            const torch::Tensor& messages_previous_step);

std::vector<torch::Tensor> compose_messages(
    const int& time_steps,
    const int& number_of_nodes,
    const int& number_of_node_features,
    const torch::Tensor& w_graph_node_features,
    const torch::Tensor& w_graph_neighbor_messages,
    const torch::Tensor& node_features,
    const torch::Tensor& adjacency_matrix,
    const torch::Tensor& messages_init);

torch::Tensor encode_messages(
    const int& number_of_nodes,
    const torch::Tensor& node_encoding_messages,
    const torch::Tensor& u_graph_node_features,
    const torch::Tensor& u_graph_neighbor_messages,
    const torch::Tensor& node_features,
    const torch::Tensor& adjacency_matrix,
    const torch::Tensor& messages);

#endif