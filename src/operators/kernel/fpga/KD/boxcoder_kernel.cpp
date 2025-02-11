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

#ifdef BOXCODER_OP

#include "operators/kernel/box_coder_kernel.h"
#include "operators/kernel/fpga/KD/box_coder_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool BoxCoderKernel<FPGA, float>::Init(BoxCoderParam<FPGA> *param) {
  param->OutputBox()->mutable_data<float>();
  param->OutputBox()->zynqmpTensor()->setAligned(false);
  param->OutputBox()->zynqmpTensor()->setDataLocation(zynqmp::CPU);
  return true;
}

template <>
void BoxCoderKernel<FPGA, float>::Compute(const BoxCoderParam<FPGA> &param) {
  param.InputPriorBox()->zynqmpTensor()->syncToCPU();
  param.InputPriorBoxVar()->zynqmpTensor()->syncToCPU();
  param.InputTargetBox()->zynqmpTensor()->syncToCPU();
  BoxCoderCompute<float>(param);
  param.OutputBox()->zynqmpTensor()->syncToCPU();
  // param.OutputBox()->zynqmpTensor()->saveToFile("boxcoder.txt");
}

template class BoxCoderKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
