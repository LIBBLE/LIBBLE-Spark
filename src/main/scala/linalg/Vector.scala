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

import java.util

/**
  * This is the trait of Vector.
  */
sealed trait Vector extends Serializable {


  /**
    * Get the i-th element
    *
    * @param i
    * @return double
    */
  def apply(i: Int): Double

  /** Length
    *
    * @return number of elements
    */
  def size: Int

  /**
    * Return a copy of this.
    *
    * @return new copy
    */
  def copy: Vector

  /**
    * Apply function on each item.
    *
    * @param f
    */
  def foreachActive(f: (Int, Double) => Unit)

  /**
    * Return the number of nonzero elements.
    *
    * @return nnz
    */
  def nnz: Int

  /**
    * Convert the vector to an array.
    *
    * @return array
    */
  def toArray: Array[Double]


}

/**
  * Class of Dense Vector.
  * @param values
  */
case class DenseVector(val values: Array[Double]) extends Vector {

  /**
    * Initialize a DenseVector with all elements zero.
    *
    * @param size
    * @return
    */
  def this(size: Int) = this {
    val temp = new Array[Double](size)
    util.Arrays.fill(temp, 0, size, 0.0)
    temp
  }

  /**
    * Return the i-th element.
    *
    * @param i
    * @return double
    */
  override def apply(i: Int): Double = values(i)


  /**
    * Return a copy of this.
    *
    * @return new copy
    */
  override def copy: DenseVector = {
    new DenseVector(values.clone())
  }

  /**
    * Return a copy of this vector.
    *
    * @return copy
    */
  override def clone(): DenseVector = {
    copy
  }

  /**
    * Return the hashcode of this vector.
    *
    * @return
    */
  override def hashCode(): Int = {
    var code = 0
    var offset = 0
    while (offset < 7) {
      val bits = java.lang.Double.doubleToLongBits(values(offset))
      code = code * 13 + (bits ^ (bits >>> 32)).toInt
      offset += 1
    }
    code
  }

  /**
    * Return the number of nonzero elements.
    *
    * @return nnz
    */
  override def nnz: Int = {
    var num = 0
    var offset = 0
    while (offset < values.length) {
      if (values(offset) != 0)
        num += 1
      offset += 1
    }
    num
  }

  /** Length
    *
    * @return number of elements
    */
  override def size: Int = values.length

  /**
    * Convert the vector to an array.
    *
    * @return array
    */
  override def toArray: Array[Double] = values

  /**
    * Convert this vector to a string.
    *
    * @return
    */
  override def toString(): String = {
    values.mkString("[", ",", "]")
  }

  /**
    * Apply function on each item.
    *
    * @param f
    */
  override def foreachActive(f: (Int, Double) => Unit): Unit = {
    var offset = 0
    while (offset < size) {
      f(offset, values(offset))
      offset += 1
    }
  }
}

/**
  * Class of the Sparse Vector.
  * @param indices
  * @param values
  * @param dim
  */
case class SparseVector(val indices: Array[Int], val values: Array[Double], dim: Int) extends Vector {
  require(indices.length == values.length && indices.length <= size, "length of indices doesn't match actual !")


  /**
    * Return the active size of element.
    *
    * @return active size
    */
  def activeSize: Int = indices.length

  /**
    * get the i-th element of this vector.
    *
    * @param i
    * @return double
    */
  override def apply(i: Int): Double = {
    var offset = 0
    while (indices(offset) < i) {
      offset += 1
    }
    if (indices(offset) == i) {
      values(offset)
    } else {
      0.0
    }
  }

  /**
    * Return a copy of this.
    *
    * @return new copy
    */
  override def copy(): SparseVector = {
    new SparseVector(indices.clone(), values.clone(), dim)
  }

  /**
    * Return a copy of this vector.
    *
    * @return copy
    */
  override def clone(): SparseVector = {
    copy()
  }

  /**
    * Return the hashcode of this vector.
    *
    * @return Int hashcode
    */
  override def hashCode(): Int = {
    var code = size * indices.length
    var offset = 0
    while (offset < values.size &&offset<7) {
      val bits = java.lang.Double.doubleToLongBits(values(offset))
      code = code * 13 + indices(offset) * (bits ^ (bits >>> 32)).toInt
      offset += 1
    }
    code
  }

  /**
    * Return the number of nonzero elements.
    *
    * @return nnz
    */
  override def nnz: Int = {
    var num = 0
    var offset = 0
    while (offset < values.length) {
      if (values(offset) != 0)
        num += 1
      offset += 1
    }
    num
  }

  /**
    * Length
    *
    * @return number of elements
    */
  override def size: Int = dim

  /**
    * Convert the vector to an array.
    *
    * @return array
    */
  override def toArray: Array[Double] = {
    val data = new Array[Double](size)
    util.Arrays.fill(data, 0, size, 0.0)
    var offset = 0
    while (offset < activeSize) {
      data(indices(offset)) = values(offset)
      offset += 1
    }
    data
  }

  /**
    * Convert the vector to a string.
    *
    * @return string
    */
  override def toString: String = {
    s"$size,${indices.mkString("[", ",", "]")},${values.mkString("[", ",", "]")}"
  }

  /**
    * Apply function on each item.
    *
    * @param f
    */
  override def foreachActive(f: (Int, Double) => Unit): Unit = {
    var offset = 0
    while (offset < activeSize) {
      f(indices(offset), values(offset))
      offset += 1
    }
  }
}