#include <torch/extension.h>
#include "../utils/derivatives.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ scalar_t compose_messages_kernel(
    const scalar_t* __restrict__ number_of_nodes,
    const scalar_t* __restrict__ previous_messages,
    const scalar_t* __restrict__ w_graph_neighbor_messages,
    const scalar_t* __restrict__ all_neighbors,
    scalar_t* __restrict__ new_messages) {

    const int index = threadIdx.x;
    const int stride = blockDim.x;

    for (int node_id = index; node_id < number_of_nodes; node_id += stride) {
      for (int end_node_index = 0; end_node_index<all_neighbors[node_id].sizes()[0]; end_node_index++){
        auto end_node_id = all_neighbors[node_id][end_node_index].item<int64_t>();
        if (end_node_id >= 0) {
          for (int neighbor_index = 0; neighbor_index<all_neighbors[node_id].sizes()[0]; neighbor_index++) {
            auto neighbor = all_neighbors[node_id][neighbor_index].item<int64_t>();
            if (neighbor >= 0 && neighbor_index!=end_node_index) {
              new_messages[node_id][end_node_id] += at::matmul(w_graph_neighbor_messages, at::relu(previous_messages[neighbor][node_id]));
            }
          }
        }
      }
    }
  return new_messages;
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

std::vector<at::Tensor> forward_cuda_cpp(
    const int& time_steps,
    const int& number_of_nodes,
    const int& number_of_node_features,
    const int& fully_connected_layer_output_size,
    const int& batch_size,
    const at::Tensor& node_features,
    const at::Tensor& all_neighbors,
    const at::Tensor& w_graph_node_features,
    const at::Tensor& w_graph_neighbor_messages,
    const at::Tensor& u_graph_node_features,
    const at::Tensor& u_graph_neighbor_messages,
    const at::Tensor& linear_weight,
    const at::Tensor& linear_bias) {
      
    auto outputs = at::zeros({batch_size, fully_connected_layer_output_size});
    auto linear_outputs = at::zeros({batch_size, fully_connected_layer_output_size});
    auto messages = at::zeros({batch_size, number_of_nodes, number_of_nodes, number_of_node_features});
    auto messages_previous_step = at::zeros({batch_size, number_of_nodes, number_of_nodes, number_of_node_features});
    auto node_encoding_messages = at::zeros({batch_size, number_of_nodes, number_of_node_features});
    auto encodings = at::zeros({batch_size, number_of_nodes*number_of_node_features});
    
    const int threads = 1024;
    const int blocks = std::floor(number_of_nodes/threads) + 1;
      
    for (int batch = 0; batch<batch_size; batch++) {
      auto new_messages = at::zeros_like({messages[batch]});
      auto previous_messages = at::zeros_like({messages[batch]});
      auto base_messages = at::matmul(w_graph_node_features, node_features);
      
      for (int time_step = 0; time_step<time_steps; time_step++) {
        std::swap(messages_previous_step, new_messages);
        AT_DISPATCH_FLOATING_TYPES(new_messages.type(), "forward_cpp_cuda", ([&] {
          compose_messages_kernel<scalar_t><<<blocks, threads>>>(number_of_nodes.data<scalar_t>(),
                                          previous_messages.data<scalar_t>(),
                                          w_graph_neighbor_messages.data<scalar_t>(),
                                          all_neighbors[batch].data<scalar_t>(),
                                          new_messages.data<scalar_t>());
                                      }));
        new_messages += base_messages;
                                    }

      messages[batch] = new_messages;
      messages_previous_step[batch] = previous_messages;
      encodings[batch] = encode_messages(number_of_nodes,
                                        node_encoding_messages[batch],
                                        u_graph_node_features,
                                        u_graph_neighbor_messages,
                                        node_features[batch],
                                        all_neighbors[batch],
                                        messages[batch]).view({-1});
      linear_outputs[batch] = at::add(at::matmul(linear_weight, encodings[batch]), linear_bias);
      outputs[batch] = at::sigmoid(linear_outputs[batch]);
    }
    return {outputs, linear_outputs, encodings, messages, messages_previous_step};
  }

std::vector<at::Tensor> backward_cpp(
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
  
  auto delta_1 = grad_output*d_sigmoid(linear_outputs);
  auto d_linear_bias = delta_1;
  auto d_linear_weight = at::matmul(delta_1.transpose(0, 1), encodings);
  
  auto delta_2 = at::matmul(delta_1, linear_weight).reshape({batch_size.item<int>(), number_of_nodes.item<int>(), number_of_node_features.item<int>()})*(d_relu_2d(encodings).reshape({batch_size.item<int>(), number_of_nodes.item<int>(), number_of_node_features.item<int>()}));
  auto d_u_graph_node_features = at::matmul(delta_2, node_features.transpose(1, 2));
  auto d_u_graph_neighbor_messages = at::matmul(delta_2.transpose(1, 2), messages_summed);

  auto delta_3 = at::matmul(delta_2.transpose(1, 2), at::matmul(u_graph_neighbor_messages_summed, d_relu_4d(messages).transpose(2, 3)));
  auto d_w_graph_node_features = at::matmul(delta_3.transpose(1, 2), node_features.transpose(1, 2));
  auto d_w_graph_neighbor_messages = at::matmul(delta_3, messages_previous_step_summed);


  return {d_w_graph_node_features, 
          d_w_graph_neighbor_messages, 
          d_u_graph_node_features, 
          d_u_graph_neighbor_messages,
          d_linear_weight,
          d_linear_bias};
}