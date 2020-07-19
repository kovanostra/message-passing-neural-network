#ifndef ARRAY_OPERATIONS_H
#define ARRAY_OPERATIONS_H

#include <torch/extension.h>

torch::Tensor remove_element_by_index_from_tensor(const torch::Tensor& tensor, const int& index);

#endif