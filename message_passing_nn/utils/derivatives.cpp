#include <torch/extension.h>
#include "../utils/derivatives.h"

at::Tensor d_sigmoid(at::Tensor z) {
  auto s = at::sigmoid(z);
  return (1 - s) * s;
}

at::Tensor d_relu_2d(at::Tensor z) {
  auto output = at::zeros_like(z);
  for (int i = 0; i<z.sizes()[0]; i++) {
    for (int j = 0; j<z.sizes()[1]; j++) {
      if (z[i][j].item<float>() > 0.0) {
        output[i][j] = 1;
      } 
    }
  }
  return output;
}

at::Tensor d_relu_4d(at::Tensor z) {
  auto output = at::zeros_like(z);
  for (int i = 0; i<z.sizes()[0]; i++) {
    for (int j = 0; j<z.sizes()[1]; j++) {
      for (int k = 0; j<z.sizes()[1]; j++) {
        for (int l = 0; j<z.sizes()[1]; j++) {
          if (z[i][j][k][l].item<float>() > 0.0) {
            output[i][j][k][l] = 1;
          }
        }
      } 
    }
  }
  return output;
}
