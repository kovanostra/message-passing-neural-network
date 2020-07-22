#ifndef DERIVATIVES_H
#define DERIVATIVES_H

#include <torch/extension.h>

at::Tensor d_sigmoid(at::Tensor z);
at::Tensor d_relu_2d(at::Tensor z);
at::Tensor d_relu_4d(at::Tensor z);

#endif