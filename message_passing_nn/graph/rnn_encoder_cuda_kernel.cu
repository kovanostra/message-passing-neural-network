#include <torch/extension.h>
#include "../utils/derivatives.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void compose_messages_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> base_neighbor_messages,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> w_graph_neighbor_messages,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> all_neighbors,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_messages) {

    const int index = threadIdx.x;
    const int stride = blockDim.x;

    for (int node_id = index; node_id < all_neighbors.size(0); node_id += stride) {
      for (int end_node_index = 0; end_node_index < all_neighbors.size(1); end_node_index++){
        auto end_node_id = std::round(all_neighbors[node_id][end_node_index]);
        if (end_node_id >= 0) {
          for (int neighbor_index = 0; neighbor_index < all_neighbors.size(1); neighbor_index++) {
            auto neighbor = std::round(all_neighbors[node_id][neighbor_index]);
            if (neighbor >= 0 && neighbor_index!=end_node_index) {
              new_messages[node_id][end_node_id] += base_neighbor_messages[neighbor][node_id];
            }
          }
        }
      }
    }
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
      
    auto outputs = at::zeros({batch_size.item<int>(), fully_connected_layer_output_size.item<int>()}), at::kCUDA);
    auto linear_outputs = at::zeros({batch_size.item<int>(), fully_connected_layer_output_size.item<int>()}, at::kCUDA);
    auto messages = at::zeros({batch_size.item<int>(), number_of_nodes.item<int>(), number_of_nodes.item<int>(), number_of_node_features.item<int>()}, at::kCUDA);
    auto messages_previous_step = at::zeros({batch_size.item<int>(), number_of_nodes.item<int>(), number_of_nodes.item<int>(), number_of_node_features.item<int>()}, at::kCUDA);
    auto node_encoding_messages = at::zeros({batch_size.item<int>(), number_of_nodes.item<int>(), number_of_node_features.item<int>()}, at::kCUDA);
    auto encodings = at::zeros({batch_size.item<int>(), number_of_nodes.item<int>()*number_of_node_features.item<int>()}, at::kCUDA);
    
    const int threads = 1024;
    const dim3 blocks(std::floor(number_of_nodes.item<int>()/threads) + 1);
      
    for (int batch = 0; batch<batch_size.item<int>(); batch++) {
      auto new_messages = at::zeros_like({messages[batch]}, at::kCUDA);
      auto previous_messages = at::zeros_like({messages[batch]}, at::kCUDA);
      auto base_messages = at::matmul(w_graph_node_features, node_features, at::kCUDA);
      const auto number_of_nodes = all_neighbors[batch].size(0);
      const auto max_neighbors = all_neighbors[batch].size(1);
      
      for (int time_step = 0; time_step<time_steps.item<int>(); time_step++) {
        auto base_neighbor_messages = at::matmul(w_graph_neighbor_messages, at::relu(previous_messages, at::kCUDA), at::kCUDA);
        std::swap(messages_previous_step, new_messages);
        auto neighbors_of_batch = all_neighbors[batch];
        AT_DISPATCH_FLOATING_TYPES(new_messages.type(), "forward_cpp_cuda", ([&] {
          compose_messages_kernel<scalar_t><<<blocks, threads>>>(base_neighbor_messages.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                 w_graph_neighbor_messages.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                 neighbors_of_batch.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                                                 new_messages.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
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
      linear_outputs[batch] = at::add(at::matmul(linear_weight, encodings[batch], at::kCUDA), linear_bias, at::kCUDA);
      outputs[batch] = at::sigmoid(linear_outputs[batch], at::kCUDA);
    }
    return {outputs, linear_outputs, encodings, messages, messages_previous_step};
  }

std::vector<at::Tensor> backward_cuda_cpp(
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
  auto d_w_graph_neighbor_messages = at::matmul(delta_3.transpose(1, 2), messages_previous_step_summed.transpose(1, 2));


  return {d_w_graph_node_features, 
          d_w_graph_neighbor_messages, 
          d_u_graph_node_features, 
          d_u_graph_neighbor_messages,
          d_linear_weight,
          d_linear_bias};
}