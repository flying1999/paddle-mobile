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
#ifdef FUSION_FC_OP

#include "fpga/KD/pes/fully_connected_pe.hpp"
#include "operators/kernel/fc_relu_kernel.h"

namespace paddle_mobile {
namespace operators {

using FullyConnectedPE = zynqmp::FullyConnectedPE;

template <>
bool FusionFcReluKernel<FPGA, float>::Init(FusionFcReluParam<FPGA>* param) {
  param->Out()->mutable_data<half>();

  FullyConnectedPE& pe = param->context().pe<FullyConnectedPE>();
  zynqmp::FullyConnectedParam& fc_param = pe.param();
  fc_param.input = param->InputX()->zynqmpTensor();
  fc_param.output = param->Out()->zynqmpTensor();
  fc_param.filter = param->InputY()->zynqmpTensor();
  fc_param.bias = param->InputZ()->zynqmpTensor();
  fc_param.relu.enabled = true;
  pe.init();
  pe.apply();
  return true;
}

template <>
void FusionFcReluKernel<FPGA, float>::Compute(
    const FusionFcReluParam<FPGA>& param) {
  zynqmp::Context& context = const_cast<zynqmp::Context&>(param.context_);
  FullyConnectedPE& pe = context.pe<FullyConnectedPE>();
  pe.dispatch();
  // param.Out()->zynqmpTensor()->saveToFile("fc_relu", true);

#ifdef PADDLE_MOBILE_DEBUG
  zynqmp::Debugger::get_instance().registerOutput("fusion_fc_relu",
                                                  param.Out()->zynqmpTensor());
#endif
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
