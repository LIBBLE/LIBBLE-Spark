package libble.classification

import libble.generalizedLinear.{GeneralizedLinearModel, HingeLoss, L2Reg}

/**
  *
  * @param stepSize
  * @param regParam
  * @param factor
  * @param iters
  * @param partsNum
  */
class SVM(stepSize: Double,
          regParam: Double,
          factor: Double,
          iters: Int,
          partsNum: Int) extends GeneralizedLinearModel(stepSize, regParam, factor, iters, partsNum){
  setLossFunc(new HingeLoss)
  setRegularizer(new L2Reg)
  setThreshold(0.0)

}
