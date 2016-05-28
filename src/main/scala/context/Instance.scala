package libble.context

import libble.linalg.Vector

/**
  * This class is used to denote one term of the training or testing data, which is consisted of
  * one label and one Vector.
  * @param label
  * @param features
  */
case class Instance(val label: Double, val features: Vector) {
  override def toString: String = s"($label, $features)"
}
