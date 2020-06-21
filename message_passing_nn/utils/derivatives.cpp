#include <torch/extension.h>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

torch::Tensor d_relu_2d(torch::Tensor z) {
  auto output = torch::zeros_like(z);
  for (int i = 0; i<z.sizes()[0]; i++) {
    for (int j = 0; j<z.sizes()[1]; j++) {
      if (z[i][j].item<float>() > 0.0) {
        output[i][j] = 1;
      } 
    }
  }
  return output;
}

torch::Tensor d_relu_4d(torch::Tensor z) {
  auto output = torch::zeros_like(z);
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
