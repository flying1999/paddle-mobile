/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include "../float16.hpp"
#include "../pe.hpp"
#include "../pe_params.hpp"
#include "conv_process.hpp"

namespace paddle_mobile {
namespace zynqmp {

class DepthwiseConvPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    DepthwiseConvParam& param = param_;
    Tensor* input = param.input;
    Tensor* output = param.output;
    int channel = output->shape().channel();
    
    float16* b_data = bias_.mutableData<float16>(FP16, param_.bias()->shape());
    if (param_.bias()->dataType() == FP32) {
      float* new_bias_data = param_.bias()->data<float>();
      // bias从float转换成float16   
      for (int i = 0; i < channel; i++) {
        b_data[i] = float_to_half(new_bias_data[i]);
      }
      bias_.flush();
    } else {
      float16* new_bias_data = param_.bias()->data<float16>();
      memcpy(b_data, new_bias_data, channel * sizeof(float16));
      bias_.flush();
    }

    if (param_.scale()->dataType() == FP32) {
      float* new_scale_data = param_.scale()->data<float>();
      Tensor* quantized_filter = param.quantizedFilter();
      quantized_filter->mutableData<float16>(FP16, param.filter->shape());
      format_dw_filter(param.filter, param.quantizedFilter(), new_scale_data);
      
    } else {
      //TODO filter 全为1时，且channal为对齐时
      float16* scale_data = param_.scale()->data<float16>();
      float16* filter_data = param.quantizedFilter()->mutableData<float16>(FP16, param.filter->shape());
      // memcpy(filter_data, scale_data, channel * sizeof(float16));
      memcpy(filter_data, scale_data, param.filter->shape().numel() * sizeof(float16));
      param.quantizedFilter()->flush();
    }
   

    DWconvArgs args = {0};
    args.bias_address = b_data;
    args.filter_address = param.quantizedFilter()->data<void>();
    args.kernel.width = param.filter->shape().height();
    args.kernel.height = param.filter->shape().width();
    args.kernel.stride_w = param.strides[0];
    args.kernel.stride_h = param.strides[1];
    args.image.address = input->data<void>();
    args.image.channels = input->shape().channel();
    args.image.height = input->shape().height();
    args.image.width = input->shape().width();
    args.image.pad_width = param.paddings[0];
    args.image.pad_height = param.paddings[1];
    args.image.scale_address = input->scale();
    args.output.address = output->data<void>();
    args.output.scale_address = output->scale();
    args.out_width = param.output->shape().width();
    args.out_height = param.output->shape().height();
    args.sub_conv_num = 1;
    param.args = args;

    inplace_.relu_enable = param_.relu.enabled;
    inplace_.power_enable = false;
    inplace_.normalize_enable = false;
  }

  bool dispatch() {
    param_.input->syncToDevice();
    if (param_.relu.enabled) {
      inplace_.relu_enable = param_.relu.enabled;
      config_inplace(inplace_);
    }
    bool ret = compute_fpga_dwconv(param_.args) == 0;
    if (param_.relu.enabled) {
      inplace_.relu_enable = false;
      config_inplace(inplace_);
    }
    return ret;
  }

  DepthwiseConvParam& param() { return param_; }

 private:
  DepthwiseConvParam param_;
  Tensor bias_;
  InplaceArgs inplace_ = {0};
};

}  // namespace zynqmp
}  // namespace paddle_mobile
