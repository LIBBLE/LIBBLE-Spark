/**
  * Created by syh on 2016/12/9.
  */
package libble.utils

import scala.collection.mutable.{Map => mutableMap}

class WorkerStore() {
  val store = mutableMap[String, Any]()

  def get[T](key: String): T = {
    store(key).asInstanceOf[T]
  }

  def put(key: String, value: Any) = {
    store += (key -> value)
  }
}