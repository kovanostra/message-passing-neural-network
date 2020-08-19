#ifndef MESSAGES_H
#define MESSAGES_H

#include <torch/extension.h>

at::Tensor get_messages_to_all_end_nodes(const int& node_id,
                                            const at::Tensor& w_graph_neighbor_messages,
                                            const at::Tensor& w_graph_node_features,
                                            const at::Tensor& all_neighbors,
                                            const at::Tensor& features_of_specific_node,
                                            at::Tensor& messages_previous_step);

std::vector<at::Tensor> compose_messages(
    const int& time_steps,
    const int& number_of_true_nodes,
    const int& number_of_node_features,
    const at::Tensor& w_graph_node_features,
    const at::Tensor& w_graph_neighbor_messages,
    const at::Tensor& base_messages,
    const at::Tensor& all_neighbors,
    const at::Tensor& messages_init);

at::Tensor encode_messages(
    const int& number_of_true_nodes,
    const at::Tensor& node_encoding_messages,
    const at::Tensor& u_graph_node_features,
    const at::Tensor& u_graph_neighbor_messages,
    const at::Tensor& node_features,
    const at::Tensor& all_neighbors,
    const at::Tensor& messages);

#endif