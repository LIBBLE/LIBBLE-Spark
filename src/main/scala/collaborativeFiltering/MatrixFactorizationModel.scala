package libble.collaborativeFiltering

import libble.linalg.Vector
import libble.linalg.implicits._
import org.apache.spark.rdd.RDD

class MatrixFactorizationModel (rank: Int,
                                userFactors: RDD[(Int, Vector)],
                                itemFactors: RDD[(Int, Vector)]) extends Serializable{
  def predict (userIndex: Int, itemIndex: Int) : Double = {
    val uh = userFactors.lookup(userIndex).head
    val vj = itemFactors.lookup(itemIndex).head
    uh * vj
  }
  def predict (indices: RDD[(Int, Int)]): RDD[Rating] = {
    val numUsers = indices.keys.distinct().count()
    val numItems = indices.values.distinct().count()
    if (numUsers > numItems){
      itemFactors.join(indices.map(_.swap)).map{
        case (item, (item_factors, user)) => (user, (item, item_factors))
      }
        .join(userFactors).map{
        case (user, ((item, item_factors), user_factors)) =>
          new Rating(item_factors * user_factors, user, item)
      }
    }
    else{
      userFactors.join(indices).map{
        case (user, (user_factors, item)) => (item, (user, user_factors))
      }
        .join(itemFactors).map{
        case (item, ((user, user_factors), item_factors)) =>
          new Rating(item_factors * user_factors, user, item)
      }
    }
  }
}