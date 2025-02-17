#pragma once

#include <torch/torch.h>

int marching_cubes_33(const at::Tensor cubes, const at::Tensor cube_types, const at::Tensor cube_edge_id, at::Tensor triangles);