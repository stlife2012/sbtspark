package com.stlife.jqxx.c4

import org.apache.spark.SparkContext
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.mllib.evaluation.RegressionMetrics

import scala.collection.Map

object DataApp {
  def main(arg:Array[String]): Unit ={
//    基于用户的协同推荐
    coorUser()
  }

  def coorUser(): Unit ={
    val sc = new SparkContext("local[1]","c4")
    val data = sc.textFile("data/ml-100k/u.data")
    val rating = data.map(line=>line.split("\t")).map(line=>Rating(line(0).toInt,line(1).toInt,line(2).toDouble))
    //    val model = ALS.train(rating,3,10)
    val model = MatrixFactorizationModel.load(sc,"model/als")
    //    model.save(sc,"model/als")
    println("用户因子矩阵:" + model.userFeatures.count())
    //    model.userFeatures.foreach(println)
    println("物品因子矩阵:" + model.productFeatures.count())
    //    model.productFeatures.foreach(println)
    val userKey = 196

    val predRating = model.predict(userKey,242)
    println("预测用户196与物品242的评价：" + predRating)

    //    以用户ID为key进行排序，并查询出对应的用户值
    val userMovies = rating.keyBy(_.user).lookup(userKey)
    println("用户所评价的电影：" + userMovies.size)
    userMovies.sortBy(-_.rating).take(10).foreach(println)

    model.recommendUsers(userKey,3).foreach(println)
    val titles = getTitles(sc)
    //    println(titles(242))
    println("获取评级最高的前10部电影:")
    userMovies.sortBy(-_.rating).take(10).map(row=>(titles(row.product),row.rating)).foreach(println)

//    println(model.userFeatures.lookup(userKey).head)
//    评估预测结果
    val userProduct = rating.map(row=>(row.user,row.product))
    val upRating = rating.map(row=>((row.user,row.product),row.rating))
    val predAndActual = model.predict(userProduct).map(rating=>((rating.user,rating.product),rating.rating)).join(upRating).map{
      case ((user,product),(pred,actual)) => (pred,actual)
    }

    predAndActual.top(10).foreach(println)
    val mse = predAndActual.map{case (pred,actual)=>math.pow((pred-actual),2)}.reduce((x,y)=>x+y) / predAndActual.count()
    val eval = new RegressionMetrics(predAndActual)
    println("MSE:" + mse.formatted("%.4f"))
    println("MSE:" + eval.meanSquaredError.formatted("%.4f"))
    println("RMSE:" + eval.rootMeanSquaredError.formatted("%.4f"))
//    println(model.predict(userKey,655))
  }

  //训练模型
  def train(sc:SparkContext): Unit ={
    val data = sc.textFile("data/ml-100k/u.data")
    val rating = data.map(line=>line.split("\t")).map(line=>Rating(line(0).toInt,line(1).toInt,line(2).toDouble))
    val model = ALS.train(rating,3,5)
    model.save(sc,"model/als")
  }
  //获取ID与电影名称
  def getTitles(sc:SparkContext): Map[Int, String] ={
    val movies = sc.textFile("data/ml-100k/u.item")
    val titles = movies.map(line => line.split("\\|").take(2)).map(array
    => (array(0).toInt,array(1))).collectAsMap()
    return titles
  }
}
