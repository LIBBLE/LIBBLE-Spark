/**
  * We licence this file to you under the Apache Licence 2.0; you could get a copy
  * of the licence from http://www.apache.org/licenses/LICENSE-2.0.
  */
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
