/**
  * Copyright (c) 2016 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.

  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at

  * http://www.apache.org/licenses/LICENSE-2.0

  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License. */
package libble.generalizedLinear

import libble.linalg.{DenseVector, Vector}
import libble.linalg.implicits._

/**
  * In this class, you should give your own function's gradient.
  * We give some instances of different losses.
  * If you want to run optimization on your own function, you should extends
  * this class, and override the function compute, give the gradient or gradient factor here.
  */
abstract class LossFunc extends Serializable {
  /**
    * The gradient of a convex function is obtained by a deltafactor*x (where x is the data point).
    * Here, we return this gradient factor.
    *
    * @param data
    * @param label
    * @param weights
    * @return delta factor
    *
    */

  def deltaF(data: Vector, label: Double, weights: Vector): Vector

  /**
    * Here we return the gradient factor and loss.
    *
    * @param data
    * @param label
    * @param weights
    * @return delta cumulate to cum,and loss
    */
  def deltaFWithLoss(data: Vector, label: Double, weights: Vector): (Vector, Double)

  /**
    * Give the pridict on data with the weights
    *
    * @param data
    * @param weights
    * @return predict Result
    */
  def predict(data: Vector, weights: Vector): Double

  def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }
}


class LogisticLoss(classNum: Int) extends LossFunc {

  def this() = this(2)


  /**
    * The gradient of a convex function is obtained by a deltafactor*x (where x is the data point).
    * Here, we return this gradient factor.
    *
    * @param data
    * @param label
    * @param weights
    * @return delta factor
    *
    */
  override def deltaF(data: Vector, label: Double, weights: Vector): Vector = {
    require((data.size % weights.size) == 0 || classNum == (weights.size / data.size + 1), "weights size not match!!!")
    classNum match {
      case 2 => {
        val margin = -1.0 * (data * weights)
        val factor = (1.0 / (1.0 + math.exp(margin))) - label
        new DenseVector(Array(factor))
      }
      case _ => {
        val dim = data.size
        var marginY = 0.0
        var maxIndex = 0
        var maxMargin = Double.NegativeInfinity
        val margins = Array.tabulate(classNum - 1) { p =>
          var tMargin = 0.0
          data.foreachActive((i, v) => {
            if (v != 0) {
              tMargin += v * weights(p * dim + i)
            }
          })
          if (p == (label - 1))
            marginY = tMargin
          if (tMargin > maxMargin) {
            maxIndex = p
            maxMargin = tMargin
          }
          tMargin
        }

        val sum = {
          var temp = 0.0
          if (maxMargin > 0) {
            for (i <- 0 until classNum - 1) {
              margins(i) -= maxMargin
              if (i == maxIndex) {
                temp += math.exp(-maxMargin)
              }
              else {
                temp += math.exp(margins(i))
              }
            }
          } else {
            for (i <- 0 until classNum - 1) {
              temp += math.exp(margins(i))
            }
          }
          temp
        }
        val deltaFactor = new Array[Double](classNum - 1)

        for (i <- 0 until classNum - 1) {
          val la = {
            if (label != 0.0 && label == i + 1)
              1.0
            else
              0.0
          }
          deltaFactor(i) = math.exp(margins(i)) / (sum + 1.0) - la
        }

        new DenseVector(deltaFactor)
      }
    }
  }

  /**
    * Here we return the gradient factor and loss.
    *
    * @param data
    * @param label
    * @param weights
    * @return delta fator,and loss
    */
  override def deltaFWithLoss(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    require((data.size % weights.size) == 0 || classNum == (weights.size / data.size + 1), "weights size not match!!!")
    classNum match {
      case 2 => {
        val margin = -1.0 * (data * weights)
        val factor = (1.0 / (1.0 + math.exp(margin))) - label
        if (label > 0) {
          (new DenseVector(Array(factor)), log1pExp(margin))
        }
        else {
          (new DenseVector(Array(factor)), log1pExp(margin) - margin)
        }
      }
      case _ => {
        val dim = data.size
        var marginY = 0.0
        var maxIndex = 0
        var maxMargin = Double.NegativeInfinity
        val margins = Array.tabulate(classNum - 1) { p =>
          var tMargin = 0.0
          data.foreachActive((i, v) => {
            if (v != 0.0) {
              tMargin += v * weights(p * dim + i)
            }
          })
          if (p == (label - 1))
            marginY = tMargin
          if (tMargin > maxMargin) {
            maxIndex = p
            maxMargin = tMargin
          }
          tMargin
        }

        val sum = {
          var temp = 0.0
          if (maxMargin > 0) {
            for (i <- 0 until classNum - 1) {
              margins(i) -= maxMargin
              if (i == maxIndex) {
                temp += math.exp(-maxMargin)
              }
              else {
                temp += math.exp(margins(i))
              }
            }
          } else {
            for (i <- 0 until classNum - 1) {
              temp += math.exp(margins(i))
            }
          }
          temp
        }
        val deltaFactor = new Array[Double](classNum - 1)

        for (i <- 0 until classNum - 1) {
          val la = {
            if (label != 0.0 && label == i + 1)
              1.0
            else
              0.0
          }
          deltaFactor(i) = math.exp(margins(i)) / (sum + 1.0) - la
        }
        var loss = {
          if (label > 0.0) {
            math.log1p(sum) - marginY
          } else {
            math.log1p(sum)
          }
        }
        if (maxMargin > 0) {
          loss += maxMargin
        }

        (new DenseVector(deltaFactor), loss)
      }
    }
  }

  /**
    * Give the pridict on data with the weights
    *
    * @param data
    * @param weights
    * @return predict Result
    */
  override def predict(data: Vector, weights: Vector): Double = {
    require((data.size % weights.size) == 0 || classNum == (weights.size / data.size + 1), "weights size not match!!!")
    classNum match {
      case 2 =>
        val margin = -(data * weights)
        1.0 / log1pExp(margin)
      case _ =>
        var maxMargin = 0.0
        var softMax = 0
        val dataSize = data.size
        for (p <- 0 until classNum) {
          var margin = 0.0
          data.foreachActive((i, v) =>
            margin += v * weights(p * dataSize + i)
          )
          if (margin > maxMargin) {
            maxMargin = margin
            softMax = p
          }

        }
        softMax
    }

  }


}

/**
  *
  */
class HingeLoss extends LossFunc {
  /**
    * The gradient of a convex function is obtained by a deltafactor*x (where x is the data point).
    * Here, we return this gradient factor.
    *
    * @param data
    * @param label
    * @param weights
    * @return delta factor
    *
    */
  override def deltaF(data: Vector, label: Double, weights: Vector): Vector = {
    val innerP = data * weights
    val factor = 2 * label - 1.0
    if (1.0 > factor * innerP) {
      new DenseVector(Array(-factor))
    }
    else {
      new DenseVector(1)
    }
  }

  /**
    * Here we return the gradient factor and loss.
    *
    * @param data
    * @param label
    * @param weights
    * @return delta cumulate to cum,and loss
    */
  override def deltaFWithLoss(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val innerP = data * weights
    val factor = 2 * label - 1.0
    if (1.0 > factor * innerP) {
      (new DenseVector(Array(-factor)), 1.0 - factor * innerP)
    }
    else {
      (new DenseVector(1), 0.0)
    }
  }

  /**
    * Give the pridict on data with the weights
    *
    * @param data
    * @param weights
    * @return predict Result
    */
  override def predict(data: Vector, weights: Vector): Double = {
    data * weights
  }
}

/**
  *
  */
class LeastSquareLoss extends LossFunc {
  /**
    * The gradient of a convex function is obtained by a deltafactor*x (where x is the data point).
    * Here, we return this gradient factor.
    *
    * @param data
    * @param label
    * @param weights
    * @return delta factor
    *
    */
  override def deltaF(data: Vector, label: Double, weights: Vector): Vector = {
    new DenseVector(Array(data * weights - label))
  }

  /**
    * Here we return the gradient factor and loss.
    *
    * @param data
    * @param label
    * @param weights
    * @return delta cumulate to cum,and loss
    */
  override def deltaFWithLoss(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val deltaF = data * weights - label
    (new DenseVector(Array(deltaF)), deltaF * deltaF / 2.0)
  }

  /**
    * Give the pridict on data with the weights
    *
    * @param data
    * @param weights
    * @return predict Result
    */
  override def predict(data: Vector, weights: Vector): Double = {
    data * weights
  }
}


