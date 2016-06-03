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
package libble.linalg

import scala.collection.mutable.ArrayBuffer

/**
  * Using with the implicit method, add methods to the Vectors.
  * @param vec
  */
class VectorsOp(val vec: Vector) {
  /**
    * Compute the inner product of two vectors.
    * vec*yT
    *
    * @param y
    * @return inner product
    */

  def *(y: Vector): Double = dot(y)

  /**
    * Compute the inner product of two vectors.
    * x*yT
    *
    * @param y
    * @return inner product
    */
  def dot(y: Vector): Double = {
    require(vec.size == y.size, s"sizes not maching for x.size = ${vec.size} while y.size = ${y.size}")
    (vec, y) match {
      case (sx: SparseVector, sy: SparseVector) => dot(sx, sy)
      case (sx: SparseVector, dy: DenseVector) => dot(sx, dy)
      case (dx: DenseVector, dy: DenseVector) => dot(dx, dy)
      case (dx: DenseVector, sy: SparseVector) => dot(sy, dx)
      case _ => throw new IllegalArgumentException("dot only support SparseVector or DenseVector")
    }
  }

  /**
    *
    * Compute the inner product of two sparse vectors.
    * x*yT
    *
    * @param x
    * @param y
    * @return inner product
    */
  private def dot(x: SparseVector, y: SparseVector): Double = {
    val xIndices = x.indices
    val xValues = x.values
    val yIndices = y.indices
    val yValues = y.values
    var sum = 0.0
    var i = 0
    var j = 0
    while (i < xIndices.length && j < yIndices.length) {
      if (xIndices(i) < yIndices(j)) {
        i += 1
      }
      else if (xIndices(i) > yIndices(j)) {
        j += 1
      }
      else {
        sum += xValues(i) * yValues(j)
        i += 1
        j += 1

      }
    }
    sum
  }

  /**
    *
    * Compute the inner product of one spartse vector and one dense vector.
    * x*yT
   *
    * @param x
    * @param y
    * @return inner product
    */
  private def dot(x: SparseVector, y: DenseVector): Double = {
    val xIndices = x.indices
    val xValues = x.values
    val yValues = y.values
    var sum = 0.0
    var offset = 0
    while (offset < x.activeSize) {
      sum += xValues(offset) * yValues(xIndices(offset))
      offset += 1
    }
    sum
  }

  /**
    *
    * Compute inner product of two dense vector.
    * x*yT
   *
    * @param x
    * @param y
    * @return inner product
    */
  private def dot(x: DenseVector, y: DenseVector): Double = {
    val xValues = x.values
    val yValues = y.values
    var offset = 0
    var sum = 0.0
    while (offset < x.size) {
      sum += xValues(offset) * yValues(offset)
      offset += 1
    }
    sum
  }


  /**
    * Compute DenseVector plus DenseVector.
    *
    * @param dx
    * @param dy
    * @return result vector
    */
  private def plus(dx: DenseVector, dy: DenseVector): Vector = {
    val n = dx.size
    val re = new Array[Double](n)
    var i = 0
    while (i < n) {
      re(i) = dx(i) + dy(i)
      i += 1
    }
    new DenseVector(re)
  }


  private def plus(dx: DenseVector, sy: SparseVector): Vector = {
    val n = dx.size
    val yIndices = sy.indices
    val yValues = sy.values
    val re = new Array[Double](n)
    System.arraycopy(dx.values, 0, re, 0, n)
    var i = 0

    while (i < sy.activeSize) {
      re(yIndices(i)) += yValues(i)
      i += 1
    }
    new DenseVector(re)
  }


  private def plus(sx: SparseVector, sy: SparseVector): Vector = {
    val xIndices = sx.indices
    val xValues = sx.values
    val yIndices = sy.indices
    val yValues = sy.values
    val reIndices = new ArrayBuffer[Int]
    val reValues = new ArrayBuffer[Double]
    var i = 0;
    var j = 0;
    while (i < xIndices.size && j < yIndices.size) {
      if (xIndices(i) < yIndices(j)) {
        reIndices += xIndices(i)
        reValues += xValues(i)
        i += 1
      } else if (xIndices(i) > yIndices(j)) {
        reIndices += yIndices(j)
        reValues += yValues(j)
        j += 1
      } else {
        reIndices += xIndices(i)
        reValues += (xValues(i) + yValues(j))
        i += 1
        j += 1
      }
    }
    new SparseVector(reIndices.toArray, reValues.toArray, sx.size)
  }

  /**
    * Compute this plus y.
    *
    * @param y
    * @return new Vector
    */
  def +(y: Vector): Vector = {
    require(vec.size == y.size, "This vectors' length not match")
    (vec, y) match {
      case (dx: DenseVector, dy: DenseVector) => plus(dx, dy)
      case (dx: DenseVector, sy: SparseVector) => plus(dx, sy)
      case (sx: SparseVector, dy: DenseVector) => plus(dy, sx)
      case (sx: SparseVector, sy: SparseVector) => plus(sx, sy)

    }
  }

  private def minus(dx: DenseVector, dy: DenseVector): Vector = {
    val n = dx.size
    val re = new Array[Double](n)
    var i = 0
    while (i < n) {
      re(i) = dx(i) - dy(i)
      i += 1
    }
    new DenseVector(re)
  }

  private def minus(dx: DenseVector, sy: SparseVector): Vector = {
    val n = dx.size
    val yIndices = sy.indices
    val yValues = sy.values
    val re = new Array[Double](n)
    System.arraycopy(dx.values, 0, re, 0, n)
    var i = 0

    while (i < sy.activeSize) {
      re(yIndices(i)) -= yValues(i)
      i += 1
    }
    new DenseVector(re)
  }

  private def minus(sx: SparseVector, dy: DenseVector): Vector = {
    import implicits.vectorAdOps
    val re = new DenseVector(dy.size)
    re.plusax(-1.0, dy)
    re.plusax(1.0, sx)
    re
  }

  private def minus(sx: SparseVector, sy: SparseVector): Vector = {
    val xIndices = sx.indices
    val xValues = sx.values
    val yIndices = sy.indices
    val yValues = sy.values
    val reIndices = new ArrayBuffer[Int]
    val reValues = new ArrayBuffer[Double]
    var i = 0;
    var j = 0;
    while (i < xIndices.size && j < yIndices.size) {
      if (xIndices(i) < yIndices(j)) {
        reIndices += xIndices(i)
        reValues += xValues(i)
        i += 1
      } else if (xIndices(i) > yIndices(j)) {
        reIndices += yIndices(j)
        reValues += (-yValues(j))
        j += 1
      } else {
        reIndices += xIndices(i)
        reValues += (xValues(i) - yValues(j))
        i += 1
        j += 1
      }
    }
    new SparseVector(reIndices.toArray, reValues.toArray, sx.size)
  }

  def -(y: Vector): Vector = {
    require(vec.size == y.size, "Vectors' length not match")
    (vec, y) match {
      case (dx: DenseVector, dy: DenseVector) => minus(dx, dy)
      case (dx: DenseVector, sy: SparseVector) => minus(dx, sy)
      case (sx: SparseVector, dy: DenseVector) => minus(sx, dy)
      case (sx: SparseVector, sy: SparseVector) => minus(sx, sy)
    }
  }


  def +=(y: Vector): DenseVector = {
    plusax(1.0, y)
  }

  def -=(y: Vector): DenseVector = {
    plusax(-1.0, y)
  }


  /**
    * Add a*x to vector y
    * y += a * x
    *
    * @param a
    * @param x
    */
  def plusax(a: Double, x: Vector): DenseVector = (x, vec) match {
    case (sx: SparseVector, dy: DenseVector) => axp2y(a, sx, dy)
    case (dx: DenseVector, dy: DenseVector) => axp2y(a, dx, dy)
    case _ => throw new IllegalArgumentException(s"axp2y only surpport add " +
      s"to a DenseVector, while get y.type = ${vec.getClass} ")
  }

  def bitwisePow(p: Double): Vector = {
    val values = vec match {
      case dv: DenseVector => dv.values
      case sv: SparseVector => sv.values
      case _=> throw new IllegalArgumentException("bitwisePow should be performed on a DenseVector or SparseVector!!!")
    }
    var offset = 0
    while (offset < values.size) {
      values(offset) = math.pow(values(offset), p)
      offset += 1
    }
    vec
  }



  /**
    * Add a*x to vector y
    * y += a * x
    *
    * @param a
    * @param x
    * @param y
    */
  private def axp2y(a: Double, x: SparseVector, y: DenseVector): DenseVector = {
    val xIndices = x.indices
    val xValues = x.values
    val yValue = y.values
    var offset = 0
    if (a == 1.0) {
      while (offset < x.activeSize) {
        yValue(xIndices(offset)) += xValues(offset)
        offset += 1
      }
    } else if (a == -1.0) {
      while (offset < x.activeSize) {
        yValue(xIndices(offset)) -= xValues(offset)
        offset += 1
      }
    } else {
      while (offset < x.activeSize) {
        yValue(xIndices(offset)) += a * xValues(offset)
        offset += 1
      }
    }
    y
  }

  /**
    * Add a*x to vector y
    * y += a * x
    *
    * @param a
    * @param x
    * @param y
    */
  private def axp2y(a: Double, x: DenseVector, y: DenseVector): DenseVector = {
    val xValues = x.values
    val yValues = y.values
    var offset = 0
    if (a == 1.0) {
      while (offset < x.size) {
        yValues(offset) += xValues(offset)
        offset += 1
      }
    } else if (a == -1.0) {
      while (offset < x.size) {
        yValues(offset) -= xValues(offset)
        offset += 1
      }
    } else {
      while (offset < x.size) {
        yValues(offset) += a * xValues(offset)
        offset += 1
      }
    }
    y
  }

  /**
    * Scale each element of vector x with factor a.
    *
    * @param a
    */
  def *=(a: Double): Vector = scal(a)

  def /=(a: Double): Vector = scal(1 / a)


  /**
    * Scale each element of vector x with factor a.
    *
    * @param a
    */
  def scal(a: Double): Vector = vec match {
    case (sx: SparseVector) => scal(a, sx)
    case (dx: DenseVector) => scal(a, dx)
  }

  /**
    * Scale each element of sparse vector x with factor a.
    *
    * @param a
    * @param x
    */
  private def scal(a: Double, x: SparseVector): SparseVector = {
    val xValues = x.values
    var offset = 0
    while (offset < x.activeSize) {
      xValues(offset) *= a
      offset += 1
    }
    x
  }

  /**
    * Scale each element of dense vector x with factor a.
    *
    * @param a
    * @param x
    */
  private def scal(a: Double, x: DenseVector): DenseVector = {
    val xValues = x.values
    var offset = 0
    while (offset < x.size) {
      xValues(offset) *= a
      offset += 1
    }
    x
  }

  /**
    * xT*y
    *
    * @param y
    * @return
    */
  def x(y: Vector): Vector = product(y)

  /**
    *
    * x^T*y
    * @param y
    * @return
    */
  def product(y: Vector): Vector = vec match {
    case dv: DenseVector => {
      y match {
        case dy: DenseVector => product(dv, dy)
        case sy: SparseVector => product(dv, sy)
        case _ => throw new IllegalArgumentException(s"product or x only support DenseVector and SparseVector ")
      }


    }
    case _ => throw new IllegalArgumentException("product or x is only support DenseVector")

  }


  private def product(dx: DenseVector, dy: DenseVector): DenseVector = {
    val xValue = dx.values
    val data = new Array[Double](dx.size * dy.size)

    val yValues = dy.values
    var i = 0
    while (i < dx.size) {
      var j = 0
      while (j < dy.size) {
        data(i * dy.size + j) = xValue(i) * yValues(j)
        j += 1
      }
      i += 1
    }
    new DenseVector(data)
  }

  private def product(dx: DenseVector, sy: SparseVector): SparseVector = {
    val indices = new Array[Int](dx.size * sy.activeSize)
    val values = new Array[Double](dx.size * sy.activeSize)
    val xValues = dx.values
    val yValues = sy.values
    val yIndices = sy.indices
    var i = 0
    while (i < dx.size) {
      var j = 0
      while (j < sy.activeSize) {
        indices(i * sy.activeSize + j) = i * sy.size + yIndices(j)
        values(i * sy.activeSize + j) = xValues(i) * yValues(j)
        j += 1
      }
      i += 1
    }
    new SparseVector(indices, values, dx.size * sy.size)
  }


  /**
    * Compute the L1 norm of the vec.
    *
    * @return
    */
  def norm1(): Double = {
    var values = Array[Double]()
    vec match {
      case dx: DenseVector => values = dx.values
      case sx: SparseVector => values = sx.values
      case _ => throw new IllegalArgumentException("norm1 only support SparseVector or DenseVector")
    }
    values.foldLeft(0.0) { (a, b) => a + b }
  }

  /**
    * Compute the L2 norm of the vec.
    *
    * @return
    */
  def norm2(): Double = {
    var values = Array[Double]()
    vec match {
      case dx: DenseVector => values = dx.values
      case sx: SparseVector => values = sx.values
      case _ => throw new IllegalArgumentException("norm1 only support SparseVector or DenseVector")
    }
    math.sqrt(values.foldLeft(0.0) { (a, b) => a + b*b })
  }


  /**
    * Append one bit to a vector with the value equals 1.
    * @return
    */
  def appendBias(): Vector = vec match {
    case dx: DenseVector =>
      val values = dx.values
      new DenseVector(values :+ 1.0)
    case sx: SparseVector =>
      val values = sx.values
      val indices = sx.indices
      val dim = sx.size
      new SparseVector(indices :+ dim, values :+ 1.0, dim + 1)
    case _ =>
      sys.error(s"The input of appendBias needs a Vector[Double], the actual input is ${vec.getClass}")
  }

}



