/**
  * We licence this file to you under the Apache Licence 2.0; you could get a copy
  * of the licence from http://www.apache.org/licenses/LICENSE-2.0.
  */
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
   * This method generates singular values matrix and right singular vectors
   *
   * @param training
   */
  def train(training: RDD[Instance]): (Array[Double], Array[Vector]) = {
    val st = Calendar.getInstance().getTimeInMillis
    val m = new GLS_Matrix_Batch(stepSize, 0.0, 0.0, iteration, parts, batchSize, K)
    m.setStopBound(bound)
    val model = m.train(training)

    /**
     *
     * v is the right singular matrix
     * singular values matrix which is square root of eigenvalues matrix
     *
     */
    for (k <- 0 to K - 1) {
      val v = model._1(k)
      val lambda = training.map(x => Math.pow(x.features * v, 2)).reduce(_ + _)
      eigenvalues.append(math.sqrt(lambda))
      eigenvectors.append(v)
    }

    println(s"time to calculate the top ${K} eigen is: " + (Calendar.getInstance().getTimeInMillis - st))
    (eigenvalues.toArray, eigenvectors.toArray)

  }

}
