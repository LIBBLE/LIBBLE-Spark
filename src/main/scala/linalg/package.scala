package libble.linalg

/**
  *
  */
package object implicits {
  implicit def vectorAdOps(vec: Vector) = new VectorsOp(vec)
}
