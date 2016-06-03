/**
 * Copyright (c) 2016 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
 * All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */
package libble.dimReduction

import scala.collection.mutable.ArrayBuffer
import java.util.Calendar
import libble.context.Instance
import libble.linalg.{DenseVector, Vector}
import libble.linalg.implicits._
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

/**
 *
 * This class is the model of PCA
 *
 * @param K
 * @param bound
 * @param stepSize
 * @param iteration
 * @param parts
 * @param batchSize
 */

class PCA(var K: Int,
           var bound: Double,
           var stepSize: Double,
           var iteration: Int,
           var parts: Int,
           var batchSize: Int) extends Logging with Serializable {
  require(K >= 1, s"K is the number of principal components, it should be that K >= 1 but was given $K")

  var eigenvalues = new ArrayBuffer[Double]()
  var eigenvectors = new ArrayBuffer[Vector]()


  /**
   *
   * This method generates the K principle components and their relating eigenvalues
   *
   * @param training
   */
  def train(training: RDD[Instance]): (Array[Double], Array[Vector]) = {

    require(K <= training.first().features.size,
      s"data dimension size is ${training.first().features.size}, it must be greater than K=$K")

    val centerData = centralize(training)

    val st = Calendar.getInstance().getTimeInMillis
    val m = new GLS_Matrix_Batch(stepSize, 0.0, 0.0, iteration, parts, batchSize, K)
    m.setStopBound(bound)
    val model = m.train(centerData)

    /**
     * v is the kth principle components
     * lambda is the kth largest eigenvalues corresponding to v
     */
    for (k <- 0 to K - 1) {
      val v = model._1(k)
      val lambda = (1.0 / (centerData.count() - 1)) * centerData.map(x => Math.pow(x.features * v, 2)).reduce(_ + _)
      eigenvalues.append(lambda)
      eigenvectors.append(v)
    }

    println(s"time to calculate the top ${K} eigen is: " + (Calendar.getInstance().getTimeInMillis - st))
    (eigenvalues.toArray, eigenvectors.toArray)

  }

  /**
   *
   * This method centralizes raw data which is the first step of PCA
   *
   * @param data
   *
   */
  def centralize(data:RDD[Instance]): RDD[Instance] = {
    val count = data.count()
    val numF = data.first().features.size
    val average = data.treeAggregate(new DenseVector(numF))(
      seqOp = (c, v) => {
        c += v.features
        c
      }, combOp = (c1, c2) => {
        c2 += c1
        c2
      }
    )
    average /= count
    val aver = data.context.broadcast(average)

    val panedData = data.map { e =>
      val newFeatures = new DenseVector(e.features.toArray)
      newFeatures -= aver.value
      new Instance(e.label, newFeatures)
    }
    panedData
  }


  /**
   *
   * This method projects raw data to new feature space using principle components
   *
   * @param rawData
   * @param pc
   *
   */
  def transform(rawData:RDD[Instance], pc:Array[Vector]): RDD[Instance] = {
    val projected = rawData.map{ ins =>
      val arr = new ArrayBuffer[Double]()
      for(k <- pc.indices) {
        arr.append(ins.features * pc(k))
      }
      Instance(ins.label, new DenseVector(arr.toArray))
    }
    projected
  }

}
