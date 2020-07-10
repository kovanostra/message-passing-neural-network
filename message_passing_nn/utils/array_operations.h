#ifndef ARRAY_OPERATIONS_H
#define ARRAY_OPERATIONS_H

#include <torch/extension.h>

std::vector<int> find_nonzero_elements(const torch::Tensor& tensor);
int find_index_by_value(const std::vector<int>& vector, const int& value);
std::vector<int> remove_element_by_index_from_vector(std::vector<int>& vector, int& index);

#endif