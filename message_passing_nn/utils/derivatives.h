#include <torch/extension.h>

torch::Tensor d_sigmoid(torch::Tensor z);
torch::Tensor d_relu_2d(torch::Tensor z);
torch::Tensor d_relu_4d(torch::Tensor z);