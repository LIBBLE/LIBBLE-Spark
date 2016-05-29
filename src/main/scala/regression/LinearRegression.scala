/**
  * We licence this file to you under the Apache Licence 2.0; you could get a copy
  * of the licence from http://www.apache.org/licenses/LICENSE-2.0.
  */
package libble.regression

import libble.generalizedLinear.{GeneralizedLinearModel, L1Reg, LeastSquareLoss}

/**
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
                       partsNum: Int) extends GeneralizedLinearModel(stepSize, regParam, factor, iters, partsNum) {

  setLossFunc(new LeastSquareLoss)
  setRegularizer(new L1Reg())
  clearThreshold


}
