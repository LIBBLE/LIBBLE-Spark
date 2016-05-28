package libble.classification

import libble.generalizedLinear.{GeneralizedLinearModel, L2Reg, LogisticLoss}

/**
  *
  * @param stepSize
  * @param regParam
  * @param factor
  * @param iters
  * @param partsNum
  */
class LogisticRegression(stepSize: Double,
                         regParam: Double,
                         factor: Double,
                         iters: Int,
                         partsNum: Int) extends GeneralizedLinearModel(stepSize, regParam, factor, iters, partsNum) {

  setLossFunc(new LogisticLoss())
  setRegularizer(new L2Reg())
  setThreshold(0.5)


}
