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
package libble.matrixDecomposition

import scala.collection.mutable.ArrayBuffer
import java.util.Calendar

import libble.context.Instance
import libble.linalg.{DenseVector, Vector}
import libble.linalg.implicits._
import libble.dimReduction.GLS_Matrix_Batch

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD


/**
  * This is the model of SVD
  *
  * @param K
  * @param bound
  * @param stepSize
  * @param iteration
  * @param parts
  * @param batchSize
  */

class SVD(var K: Int,
          var bound: Double,
          var stepSize: Double,
          var iteration: Int,
          var parts: Int,
          var batchSize: Int) extends Logging with Serializable {
  var eigenvalues = new ArrayBuffer[Double]()
  var eigenvectors = new ArrayBuffer[Vector]()


  /**
   *
   * This method generates singular values matrix and right singular vectors.
   *
   * @param training
   */
  def train(training: RDD[Vector]): (Array[Double], Array[Vector]) = {
    val st = Calendar.getInstance().getTimeInMillis
    val m = new GLS_Matrix_Batch(stepSize, 0.0, 0.0, iteration, parts, batchSize, K)
    m.setStopBound(bound)
    val model = m.train(training)

    /**
     *
     * v is the right singular matrix
     * Singular values matrix which is square root of eigenvalues matrix.
     *
     */
    for (k <- 0 to K - 1) {
      val v = model._1(k)
      val lambda = training.map(x => Math.pow(x * v, 2)).reduce(_ + _)
      eigenvalues.append(math.sqrt(lambda))
      eigenvectors.append(v)
    }

    println(s"time to calculate the top ${K} eigen is: " + (Calendar.getInstance().getTimeInMillis - st))
    (eigenvalues.toArray, eigenvectors.toArray)

  }

}
