/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef FUSION_CONVBNLEAKYRELU_OP

#include <vector>
#include "framework/ddim.h"
#include "framework/operator.h"
#include "operators/math/im2col.h"
#include "operators/math/math_function.h"
#include "operators/math/vol2col.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

using framework::DDim;
using framework::OpKernelBase;

template <typename DeviceType, typename T>
class ConvBNLeakyReluKernel
    : public OpKernelBase<DeviceType, FusionConvBNLeakyReluParam<DeviceType>> {
 public:
  void Compute(const FusionConvBNLeakyReluParam<DeviceType> &param);
  bool Init(FusionConvBNLeakyReluParam<DeviceType> *param);
};

}  // namespace operators
}  // namespace paddle_mobile

#endif
