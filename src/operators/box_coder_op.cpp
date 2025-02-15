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

#include "operators/box_coder_op.h"
#include <vector>
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void BoxCoderOp<Dtype, T>::InferShape() const {
  auto input_priorbox_dims = this->param_.InputPriorBox()->dims();
  auto input_priorboxvar_dims = this->param_.InputPriorBoxVar()->dims();
  auto input_targetbox_dims = this->param_.InputTargetBox()->dims();

  auto code_type = this->param_.CodeType();

  if (code_type == "encode_center_size") {
    if (input_targetbox_dims.size() != 2) {
      LOG(kLOG_ERROR) << " The rank of Input of TargetBox must be 2";
    }
    if (input_targetbox_dims[1] != 4) {
      LOG(kLOG_ERROR) << " The shape of TargetBox is [M, 4]";
    }
  }
  if (code_type == "decode_center_size") {
    if (input_targetbox_dims.size() != 3) {
      LOG(kLOG_ERROR) << "The rank of Input of TargetBox must be 3";
    }
    if (input_targetbox_dims[1] != input_priorbox_dims[0] ||
        input_targetbox_dims[2] != input_priorbox_dims[1]) {
      LOG(kLOG_ERROR) << " dimension not match";
    }
  }
  this->param_.OutputBox()->Resize(framework::make_ddim(
      {input_targetbox_dims[0], input_priorbox_dims[0], 4}));
}

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(box_coder, ops::BoxCoderOp);
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(box_coder, ops::BoxCoderOp);
#endif
#if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
REGISTER_OPERATOR_FPGA(box_coder, ops::BoxCoderOp);
#endif

#endif
