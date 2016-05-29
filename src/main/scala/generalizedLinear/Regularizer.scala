/**
  * We licence this file to you under the Apache Licence 2.0; you could get a copy
  * of the licence from http://www.apache.org/licenses/LICENSE-2.0.
  */
package libble.generalizedLinear

import libble.linalg.Vector
import libble.linalg.implicits._

/**
  *
  */
abstract class Regularizer extends Serializable {
  /**
    * In this method, we update the weight with weightnew= weightOld+stepSize*(gradient+regParam*  delte(regularizer)).
    * Where delta(regularizer) is the gradient of regularizer.
    *
    * @param weights
    * @param gradient
    * @param stepSize
    * @param regParam
    * @return weightNew
    */
  def update(weights: Vector, gradient: Vector, stepSize: Double, regParam: Double): Unit

  /**
    * In this method, we give the cost of the regularizer
    *
    * @param weight
    * @param regParam
    * @return regCost
    */
  def getRegVal(weight: Vector, regParam: Double): Double

}

/**
  *
  */
class withoutReg extends Regularizer {
  /**
    * In this method, we update the weight with weightnew= weightOld+stepSize*(gradient+regParam*  delte(regularizer)).
    * Where delta(regularizer) is the gradient of regularizer.
    *
    * @param weights
    * @param gradient
    * @param stepSize
    * @param regParam
    * @return weightNew
    */
  override def update(weights: Vector, gradient: Vector, stepSize: Double, regParam: Double): Unit = {
    weights.plusax(-stepSize, gradient)
  }

  /**
    * In this method, we give the cost of the regularizer
    *
    * @param weight
    * @param regParam
    * @return regCost
    */
  override def getRegVal(weight: Vector, regParam: Double): Double = {
    0.0
  }
}

/**
  *
  */
class L1Reg extends Regularizer {
  /**
    * In this method, we update the weight with weightnew= weightOld+stepSize*(gradient+regParam*  delte(regularizer)).
    * Where delta(regularizer) is the gradient of regularizer.
    *
    * @param weights
    * @param gradient
    * @param stepSize
    * @param regParam
    * @return weightNew
    */
  override def update(weights: Vector, gradient: Vector, stepSize: Double, regParam: Double): Unit = {
    weights.plusax(-stepSize, gradient)
    val reg_step = regParam * stepSize
    val weightsValues = weights.toArray
    var offset = 0
    while (offset < weights.size) {
      weightsValues(offset) = math.signum(weightsValues(offset)) * math.max(0.0, math.abs(weightsValues(offset) - reg_step))
      offset += 1
    }
  }

  /**
    * In this method, we give the cost of the regularizer
    *
    * @param weight
    * @param regParam
    * @return regCost
    */
  override def getRegVal(weight: Vector, regParam: Double): Double = {
    weight.norm1 * regParam
  }
}

/**
  *
  */
class L2Reg extends Regularizer {
  /**
    * In this method, we update the weight with weightnew= weightOld+stepSize*(gradient+regParam*  delte(regularizer)).
    * Where delta(regularizer) is the gradient of regularizer.
    *
    * @param weights
    * @param gradient
    * @param stepSize
    * @param regParam
    * @return weightNew
    */
  override def update(weights: Vector, gradient: Vector, stepSize: Double, regParam: Double): Unit = {
    weights *= (1 - stepSize * regParam)
    weights.plusax(-stepSize, gradient)
  }

  /**
    * In this method, we give the cost of the regularizer
    *
    * @param weight
    * @param regParam
    * @return regCost
    */
  override def getRegVal(weight: Vector, regParam: Double): Double = {
    val norm = weight.norm2
    0.5 * regParam * norm * norm
  }
}

