#include <torch/extension.h>
#include <iostream>

std::vector<int> find_nonzero_elements(const at::Tensor& tensor){
  std::vector<int> vector;
  for(int index=0; index<tensor.sizes()[0]; index++){
    if(tensor[index].item<int>()!=0) {
       vector.push_back(index);
      }
  }
  return vector;
}

int find_index_by_value(const std::vector<int>& vector, const int& value){
  auto vector_size = static_cast<int>(vector.size());
  for (int index = 0; index < vector_size; ++index) {
    if (vector[index]==value) return index;
  }
  return -1;
}

std::vector<int> remove_element_by_index_from_vector(std::vector<int>& vector, int& index){
  std::vector<int> final_vector;
  if (index==0) {
    final_vector.assign(vector.begin() + 1, vector.end());
  } else if ((index<static_cast<int>(vector.size()))){
    std::vector<int> first_vector;
    std::vector<int> second_vector;
    first_vector.assign(vector.begin(), vector.begin() + index);
    second_vector.assign(vector.begin() + index + 1, vector.end());
    final_vector.reserve( first_vector.size() + second_vector.size() ); 
    final_vector.insert( final_vector.end(), first_vector.begin(), first_vector.end() );
    final_vector.insert( final_vector.end(), second_vector.begin(), second_vector.end() );
  } else {
    final_vector.assign(vector.begin(), vector.end() - 1);
  }
  return final_vector; 
}

at::Tensor compose_messages(
    const int& time_steps,
    const int& number_of_nodes,
    const int& number_of_node_features,
    const at::Tensor& w_graph_node_features,
    const at::Tensor& w_graph_neighbor_messages,
    const at::Tensor& node_features,
    const at::Tensor& adjacency_matrix,
    at::Tensor& messages) {

  for (int time_step = 0; time_step<time_steps; time_step++) {
    auto new_messages = at::zeros_like({messages});

    for (int node_id = 0; node_id<number_of_nodes; node_id++) {
      auto all_neighbors = find_nonzero_elements(adjacency_matrix[node_id]);
      auto number_of_neighbors = static_cast<int>(all_neighbors.size());
      
      for (int i = 0; i < number_of_neighbors; i++){
        auto end_node_id = all_neighbors[i];
        auto messages_from_the_other_neighbors = at::zeros_like({messages[0][0]});

        if (number_of_neighbors > 1) {
          auto end_node_index = find_index_by_value(all_neighbors, end_node_id);
          auto other_neighbors = remove_element_by_index_from_vector(all_neighbors, end_node_index);      
          auto number_of_other_neighbors = static_cast<int>(other_neighbors.size());
          for (int z = 0; z < number_of_other_neighbors; ++z) {
              auto neighbor = other_neighbors[z];
              messages_from_the_other_neighbors += torch::matmul(w_graph_neighbor_messages, messages[neighbor][node_id]);
          }
        }
        new_messages[node_id][end_node_id] = torch::relu(torch::add(torch::matmul(w_graph_node_features, 
                                                                                  node_features[node_id]), 
                                                                    messages_from_the_other_neighbors));
      }
    }
    messages = new_messages;
  }
  return messages;
}

at::Tensor encode_messages(
    const int& number_of_nodes,
    const at::Tensor& node_encoding_messages,
    const at::Tensor& u_graph_node_features,
    const at::Tensor& u_graph_neighbor_messages,
    const at::Tensor& node_features,
    const at::Tensor& adjacency_matrix,
    at::Tensor& messages) {
      
    for (int node_id = 0; node_id<number_of_nodes; node_id++) {
      auto all_neighbors = find_nonzero_elements(adjacency_matrix[node_id]);
      auto number_of_neighbors = static_cast<int>(all_neighbors.size());

      for (int i = 0; i < number_of_neighbors; i++){
        auto end_node_id = all_neighbors[i];
        node_encoding_messages[node_id] += torch::matmul(u_graph_neighbor_messages, messages[end_node_id][node_id]);
      }
    }
    return torch::relu(torch::add(torch::matmul(u_graph_node_features, node_features), node_encoding_messages));
  }

at::Tensor encode(
    const int& time_steps,
    const int& number_of_nodes,
    const int& number_of_node_features,
    const at::Tensor& w_graph_node_features,
    const at::Tensor& w_graph_neighbor_messages,
    const at::Tensor& u_graph_node_features,
    const at::Tensor& u_graph_neighbor_messages,
    const at::Tensor& node_features,
    const at::Tensor& adjacency_matrix) {

    auto messages = torch::zeros({number_of_nodes, number_of_nodes, number_of_node_features});
    messages = compose_messages(
        time_steps,
        number_of_nodes,
        number_of_node_features,
        w_graph_node_features,
        w_graph_neighbor_messages,
        node_features,
        adjacency_matrix,
        messages);
    auto node_encoding_messages = torch::zeros({number_of_nodes, number_of_node_features});
    return encode_messages(
        number_of_nodes,
        node_encoding_messages,
        u_graph_node_features,
        u_graph_neighbor_messages,
        node_features,
        adjacency_matrix,
        messages);
}

std::vector<torch::Tensor> forward_cpp(
    const int& time_steps,
    const int& number_of_nodes,
    const int& number_of_node_features,
    const int& fully_connected_layer_output_size,
    const int& batch_size,
    const at::Tensor& node_features,
    const at::Tensor& adjacency_matrix,
    const at::Tensor& w_graph_node_features,
    const at::Tensor& w_graph_neighbor_messages,
    const at::Tensor& u_graph_node_features,
    const at::Tensor& u_graph_neighbor_messages,
    const at::Tensor& linear_weight,
    const at::Tensor& linear_bias) {
      
    auto outputs = torch::zeros({batch_size, fully_connected_layer_output_size});
    auto encodings = torch::zeros({batch_size, number_of_nodes, number_of_node_features});
      
    for (int batch = 0; batch<batch_size; batch++) {
      encodings[batch] = encode(time_steps,
                                      number_of_nodes,
                                      number_of_node_features,
                                      w_graph_node_features,
                                      w_graph_neighbor_messages,
                                      u_graph_node_features,
                                      u_graph_neighbor_messages,
                                      node_features[batch],
                                      adjacency_matrix[batch]);
      outputs[batch] = torch::sigmoid(
                          torch::add(linear_bias, 
                          torch::matmul(linear_weight,encodings.view({-1}))));
      
    }
    return {outputs, encodings};
  }

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

std::vector<torch::Tensor> backward_cpp(
  const at::Tensor& grad_output,
  const at::Tensor& outputs,
  const at::Tensor& encodings_flat,
  const at::Tensor& w_graph_node_features,
  const at::Tensor& w_graph_neighbor_messages,
  const at::Tensor& u_graph_node_features,
  const at::Tensor& u_graph_neighbor_messages,
  const at::Tensor& linear_weight,
  const at::Tensor& linear_bias) {

  // auto d_linear_weight = grad_output*d_sigmoid(outputs) * outputs;
  // auto d_linear_bias = grad_output*d_sigmoid(outputs);

  // auto d_u_graph_node_features = grad_output*d_sigmoid(outputs)*linear_weight;
  // auto d_u_graph_neighbor_messages = grad_output*d_sigmoid(outputs)*linear_weight;

  // auto d_w_graph_node_features = grad_output*d_sigmoid(outputs)*linear_weight;
  // auto d_w_graph_neighbor_messages = grad_output*d_sigmoid(outputs)*linear_weight;
  auto d_linear_weight = torch::matmul((grad_output*d_sigmoid(outputs)).transpose(0, 1), encodings_flat);
  auto d_linear_bias = grad_output*d_sigmoid(outputs);

  auto d_u_graph_node_features = torch::zeros_like(u_graph_node_features);
  auto d_u_graph_neighbor_messages = torch::zeros_like(u_graph_neighbor_messages);

  auto d_w_graph_node_features = torch::zeros_like(w_graph_node_features);
  auto d_w_graph_neighbor_messages = torch::zeros_like(w_graph_neighbor_messages);


  return {d_w_graph_node_features, 
          d_w_graph_neighbor_messages, 
          d_u_graph_node_features, 
          d_u_graph_neighbor_messages,
          d_linear_weight,
          d_linear_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compose_messages", &compose_messages, "RNN encoder compose messages (CPU)");
  m.def("encode_messages", &encode_messages, "RNN encoder encode messages (CPU)");
  m.def("encode", &encode, "RNN encoder encode (CPU)");
  m.def("forward", &forward_cpp, "RNN encoder forward pass (CPU)");
  m.def("backward", &backward_cpp, "RNN encoder backward pass (CPU)");
}