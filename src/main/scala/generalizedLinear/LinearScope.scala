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
package libble.generalizedLinear

import libble.context.Instance
import libble.linalg.implicits.vectorAdOps
import libble.linalg.{DenseVector, Vector}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer
import scala.math._
import scala.util.Random

/**
  * This class is the model of Generalized Linear Algorithms with default lossfunc LogisticLoss and default regularization L2Reg.
  *
  * @param stepSize the learning rate.
  * @param regParam the regParam
  * @param factor   the elastic param
  * @param iters    the number of outer loop
  * @param partsNum the number of partitions which correspond to the number of task.
  */
class LinearScope(var stepSize: Double,
                  var regParam: Double,
                  var factor: Double,
                  var iters: Int,
                  var partsNum: Int) extends Serializable {
  def this() = this(1.0, 0.0001, 0.0001, 5, -1)

  private[this] var weights: Option[Vector] = None
  private[this] var lossfunc: LossFunc = new LogisticLoss()
  private[this] var updater: Updater = new L2Updater()

  private[this] var addBias: Boolean = true


  private[this] var stopBound: Double = 0.0


  private[this] var numPredictor: Int = 1


  var threshold: Option[Double] = Some(0.5)

  /**
    *
    * @param value
    * @return this.type
    */
  def setThreshold(value: Double): this.type = {
    threshold = Some(value)
    this
  }

  /**
    *
    * @return this.type
    */
  def clearThreshold: this.type = {
    threshold = None
    this
  }

  /**
    * Set add Bias or not.
    *
    * @param value
    * @return this.type
    */
  private[this] def setBias(value: Boolean): this.type = {
    addBias = value
    this
  }

  /**
    * Set the stop bound.
    *
    * @param value
    * @return
    */
  def setStopBound(value: Double): this.type = {
    stopBound = value
    this
  }

  /**
    * Set the lossfunc
    *
    * @param loss
    * @return this
    */
  def setLossFunc(loss: LossFunc): this.type = {
    lossfunc = loss
    this
  }

  /**
    * Set the Regularizer
    *
    * @param up
    * @return this
    */
  def setUpdater(up: Updater): this.type = {
    updater = up
    this
  }

  /**
    * Set the stepSize
    *
    * @param value
    * @return this
    */
  def setStepSize_(value: Double): this.type = {
    stepSize = value
    this
  }

  /**
    * Set the factor
    *
    * @param value
    * @return this
    */
  def setFactor(value: Double): this.type = {
    factor = value
    this
  }

  /**
    * Set the iterations
    *
    * @param value
    * @return this
    */
  def setIters(value: Int): this.type = {
    iters = value
    this
  }

  /**
    * Set the RegParam
    *
    * @param value
    * @return this
    */
  def setRegParam(value: Double): this.type = {
    regParam = value
    this
  }

  /**
    * Set the data's Parts
    *
    * @param value
    * @return this
    */
  def setParts(value: Int): this.type = {
    partsNum = value
    this
  }


  /**
    * Set the classNum
    *
    * @param classNum
    * @return this
    */
  def setClassNum(classNum: Int): this.type = {
    numPredictor = classNum - 1
    this
  }

  /**
    * Training the model on training data.
    *
    * @param trainingData
    * @return lossArray
    */
  def train(trainingData: RDD[Instance]): Array[Double] = {
    val d = trainingData.first().features.size
    val initialWeights = {
      if (addBias) {
        new DenseVector((d + 1) * numPredictor)
      } else {
        new DenseVector(d * numPredictor)
      }
    }
    train(trainingData, initialWeights)
  }

  /**
    * Training on training data with initial weights.
    *
    * @param trainingData
    * @param initialWeights
    * @return lossArray
    */
  def train(trainingData: RDD[Instance], initialWeights: Vector): Array[Double] = {
    if (partsNum == (-1)) {
      partsNum = trainingData.partitions.length
    }
    val data = {
      (addBias, partsNum == trainingData.partitions.length) match {
        case (true, true) => {
          trainingData.map(e => (e.label, e.features.appendBias()))
        }
        case (true, false) => {
          trainingData.map(e => (e.label, e.features.appendBias())).coalesce(partsNum, true).cache()

        }
        case (false, true) => {
          trainingData.map(e => (e.label, e.features)).cache()
        }
        case (false, false) => {
          trainingData.map(e => (e.label, e.features)).coalesce(partsNum, true).cache()
        }
      }
    }
    runEngine(data, initialWeights)
  }

  /**
    * Optimization engine.
    *
    * @param data
    * @param initialWeights
    * @return lossArray
    */
  private[this] def runEngine(data: RDD[(Double, Vector)], initialWeights: Vector): Array[Double] = {
    val count = data.count()
    var w = initialWeights.copy
    val n = w.size
    var convergenced = false

    val startTime = System.currentTimeMillis()
    val lossArray = ArrayBuffer[Double]()
    var i = 0
    var time = 0l
    while (i < iters && !convergenced) {
      time = System.currentTimeMillis()
      val w_0 = data.context.broadcast(w)

      val (mu, totalLoss) = data.treeAggregate(new DenseVector(n), 0.0)(
        seqOp = (c, v) => {
          val fwl = lossfunc.deltaFWithLoss(v._2, v._1, w_0.value)
          c._1 += (fwl._1 x v._2)
          (c._1, c._2 + fwl._2)
        },
        combOp = (c1, c2) => {
          (c1._1 += c2._1, c1._2 + c2._2)
        }
      )
      mu *= (1.0 / count.toDouble)

      val loss = totalLoss / count.toDouble + updater.getRegVal(w, regParam)
      lossArray += loss

      println(s"loss: $loss at iters: $i, with time: ${time - startTime}")




      val lastWeights = w.copy


      w= updater.update(data:RDD[(Double,Vector)],w:Vector,mu:Vector,lossfunc:LossFunc,stepSize:Double,factor:Double,regParam:Double)



      convergenced = isConvergenced(lastWeights, w)
      i += 1


    }

    println(s"losses of the last 5 iteration are:${lossArray.takeRight(5).mkString(",")}")
    weights = Some(w)
    lossArray.toArray
  }

  /**
    *
    * @param lastWeights
    * @param thisWeights
    * @return Boolean
    */
  private[this] def isConvergenced(lastWeights: Vector, thisWeights: Vector): Boolean = {
    val temp = lastWeights.copy
    temp -= thisWeights
    stopBound * max(thisWeights.norm2(), 1.0) > lastWeights.norm2()
  }

  /**
    * Predict on the Vector using the model.
    *
    * @param v
    * @return Double
    */
  def predict(v: Vector): Double = {
    if (threshold == None) {
      predictT(v)
    }
    else if (predictT(v) > threshold.get) 1.0 else 0.0
  }

  /**
    * Predict on the data using the model.
    *
    * @param input
    * @return RDD[Double]
    */
  def predict(input: RDD[Vector]): RDD[Double] = {
    if (threshold == None) {
      predictT(input)
    } else {
      predictT(input).map(e => if (e > threshold.get) 1.0 else 0.0)
    }
  }


  private def predictT(v: Vector): Double = weights match {

    case Some(w) => {
      require(w.size == v.size, "Vector length not match with the model")
      if (addBias) {
        lossfunc.predict(v.appendBias(), w)
      }
      else {
        lossfunc.predict(v, w)
      }
    }
    case _ => {
      throw new IllegalAccessError("you should train your model first  by call the function train!")

    }
  }


  private def predictT(input: RDD[Vector]): RDD[Double] = {
    input.map(predict)
  }

}

class WeightsVector(val partA: Vector, val partB: Vector) {
  require(partA.size == partB.size)

  import libble.linalg.implicits.vectorAdOps

  var fac_a = 1.0
  var fac_b = 0.0

  def size = partA.size

  def dot(data: Vector): Double = {
    fac_a * (partA * data) + fac_b * (partB * data)
  }

  def *(data: Vector): Double = {
    dot(data)
  }

  def apply(i: Int): Double = {
    fac_a * partA(i) + fac_b * partB(i)
  }

  def merge(): WeightsVector = {
    partA *= fac_a
    partA.plusax(fac_b, partB)
    fac_a = 1.0
    fac_b = 0.0
    this
  }
  def toDense():DenseVector={
    partA *= fac_a
    partA.plusax(fac_b, partB)
  }
}