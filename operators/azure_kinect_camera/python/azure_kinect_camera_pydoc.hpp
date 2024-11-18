/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PYHOLOHUB_OPERATORS_AZURE_KINECT_CAMERA_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_AZURE_KINECT_CAMERA_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace AzureKinectCameraOp {

PYDOC(AzureKinectCameraOp, R"doc(
The `azure_kinect_camera` operator reads RGB-D from the specified input device.

)doc")

// PyAzureKinectCameraOp Constructor
PYDOC(AzureKinectCameraOp_python, R"doc(
Operator class to capture from azure kinect camera.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
allocator: ``holoscan.resources.Allocator``
    Allocator used to allocate the volume data
device_serial : str, optional
    Serial Number of the Device or ANY
capture_timeout_ms: int, optional
    Capture timeout during grab call.
name : str, optional
    The name of the operator.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : ``holoscan.core.OperatorSpec``
    The operator specification.
)doc")

}  // namespace AzureKinectCameraOp


}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_AZURE_KINECT_CAMERA_PYDOC_HPP
