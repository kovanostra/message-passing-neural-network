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

int find_index_by_value(const std::vector<int>& vector,
                        const int& value){
  auto vector_size = static_cast<int>(vector.size());
  for (int index = 0; index < vector_size; ++index) {
    if (vector[index]==value) return index;
  }
  return -1;
}

std::vector<int> remove_element_by_index_from_vector(std::vector<int>& vector,
                                                     int& index){
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
    const int& number_of_nodes,
    const at::Tensor& w_graph_node_features,
    const at::Tensor& w_graph_neighbor_messages,
    const at::Tensor& node_features,
    const at::Tensor& adjacency_matrix,
    const at::Tensor& messages) {

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
  return new_messages;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compose_messages", &compose_messages, "RNN encoder forward (CPU)");
}