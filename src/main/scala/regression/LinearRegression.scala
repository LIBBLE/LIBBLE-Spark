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
