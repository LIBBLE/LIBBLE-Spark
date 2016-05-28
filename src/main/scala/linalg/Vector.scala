package libble.linalg

import java.util

/**
  *
  */
sealed trait Vector extends Serializable {


  /**
    * get the i-th element
    *
    * @param i
    * @return double
    */
  def apply(i: Int): Double

  /** length
    *
    * @return number of elements
    */
  def size: Int

  /**
    * return a copy of this
    *
    * @return new copy
    */
  def copy: Vector

  /**
    * apply function on each item
    *
    * @param f
    */
  def foreachActive(f: (Int, Double) => Unit)

  /**
    * return the number of nonzero elements
    *
    * @return nnz
    */
  def nnz: Int

  /**
    * convert the vector to an array
    *
    * @return array
    */
  def toArray: Array[Double]


}

/**
  *
  * @param values
  */
case class DenseVector(val values: Array[Double]) extends Vector {

  /**
    * initialize a DenseVector with all elements are zero
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
    * return the i-th element
    *
    * @param i
    * @return double
    */
  override def apply(i: Int): Double = values(i)


  /**
    * return a copy of this
    *
    * @return new copy
    */
  override def copy: DenseVector = {
    new DenseVector(values.clone())
  }

  /**
    * return a copy of this vector
    *
    * @return copy
    */
  override def clone(): DenseVector = {
    copy
  }

  /**
    * return the hashcode of this vector
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
    * return the number of nonzero elements
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

  /** length
    *
    * @return number of elements
    */
  override def size: Int = values.length

  /**
    * convert the vector to an array
    *
    * @return array
    */
  override def toArray: Array[Double] = values

  /**
    * convert this vector to a string
    *
    * @return
    */
  override def toString(): String = {
    values.mkString("[", ",", "]")
  }

  /**
    * apply function on each item
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
  *
  * @param indices
  * @param values
  * @param dim
  */
case class SparseVector(val indices: Array[Int], val values: Array[Double], dim: Int) extends Vector {
  require(indices.length == values.length && indices.length <= size, "length of indices doesn't match actual !")


  /**
    * return the active size of element
    *
    * @return active size
    */
  def activeSize: Int = indices.length

  /**
    * get the i-th element of this vector
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
    * return a copy of this
    *
    * @return new copy
    */
  override def copy(): SparseVector = {
    new SparseVector(indices.clone(), values.clone(), dim)
  }

  /**
    * return a copy of this vector
    *
    * @return copy
    */
  override def clone(): SparseVector = {
    copy()
  }

  /**
    * return the hashcode of this vector
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
    * return the number of nonzero elements
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
    * length
    *
    * @return number of elements
    */
  override def size: Int = dim

  /**
    * convert the vector to an array
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
    * convert the vector to a string
    *
    * @return string
    */
  override def toString: String = {
    s"$size,${indices.mkString("[", ",", "]")},${values.mkString("[", ",", "]")}"
  }

  /**
    * apply function on each item
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