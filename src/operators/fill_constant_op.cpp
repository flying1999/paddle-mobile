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

#ifdef FILL_CONSTANT_OP

#include "operators/fill_constant_op.h"

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(fill_constant, ops::FillConstantOp);
#endif
#if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
REGISTER_OPERATOR_FPGA(fill_constant, ops::FillConstantOp);
#endif

#endif
