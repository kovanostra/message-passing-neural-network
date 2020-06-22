#include <torch/extension.h>
#include "../utils/array_operations.h"

std::vector<int> find_nonzero_elements(const torch::Tensor& tensor){
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