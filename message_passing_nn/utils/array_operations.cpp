#include <torch/extension.h>
#include "../utils/array_operations.h"

torch::Tensor remove_element_by_index_from_tensor(const torch::Tensor tensor, 
                                                  const int& index){
  return torch::cat({tensor.index({torch::indexing::Slice(0, index)}), 
                     tensor.index({torch::indexing::Slice(index, tensor.sizes()[0])})});
}