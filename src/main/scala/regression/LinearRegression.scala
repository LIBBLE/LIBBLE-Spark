/**
  * Copyright (c) 2016 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
  * All Rights Reserved.
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
  * limitations under the License. */
package libble.regression

import libble.generalizedLinear.{L1Updater, LeastSquareLoss, LinearScope}

/**
  * This is the model of LinearRegression with default regularization L1Reg.
  *
  * @param stepSize
  * @param regParam
  * @param factor
  * @param iters
  * @param partsNum
  */
class LinearRegression(stepSize: Double,
                       regParam: Double,
                       factor: Double,
                       iters: Int,
                       partsNum: Int) extends LinearScope(stepSize, regParam, factor, iters, partsNum) {

  setLossFunc(new LeastSquareLoss)
  setUpdater(new L1Updater())

  /**
    * Set the output to be the predict value.
    */
  clearThreshold


}
