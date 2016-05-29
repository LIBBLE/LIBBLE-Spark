/**
  * We licence this file to you under the Apache Licence 2.0; you could get a copy
  * of the licence from http://www.apache.org/licenses/LICENSE-2.0.
  */
package libble.linalg

/**
  *
  */
package object implicits {
  implicit def vectorAdOps(vec: Vector) = new VectorsOp(vec)
}
