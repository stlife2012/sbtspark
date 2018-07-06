package com.stlife.jqxx.c7

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD

object DataApp {
  def main(str:Array[String]): Unit ={
//    collectData()
    kmeansTrain()
  }

  def kmeansTrain(): Unit ={//调节K参数进行判定
    val feature = collectData()
    val userFeature = feature._1
    val movieFeature = feature._2
    val splitData = movieFeature.randomSplit(Array(0.8,0.2))
    val trainData = splitData(0)
    val testData = splitData(1)
//    movieFeature.take(2).foreach(l=>println(f"${l._1}%.2f ${l._2.mkString(",")}"))
    val movieData = trainData.map(l=>Vectors.dense(l._2))
    val testmovieData = movieFeature.map(l=>Vectors.dense(l._2))
    println("Data Num:" + movieFeature.count())
    movieData.cache()
    testmovieData.cache()
    val mulMetrics = Seq(2,3,5,10,20).map(k=>{
      val model = KMeans.train(movieData,k,50,3)
//      val pred = model.predict(movieData.first())
      (k,model.computeCost(testmovieData))
    })

    //随着类中心数目增加， WCSS值会出现下降，然后又开始增大。
    mulMetrics.foreach(line=>println(f"K: ${line._1} WCSS COST:${line._2}%.2f"))
  }

  def collectData(): (RDD[(Int, Array[Double])], RDD[(Int, Array[Double])]) ={//通过ALS构建隐形特征因子
    val sc = new SparkContext("local[1]","c7")
    val rdd = sc.textFile("data/ml-100k/u.data")
    val data = rdd.map(line=>{
      val split = line.split("\t")
      Rating(split(0).toInt,split(1).toInt,split(2).toDouble)
    })
    data.cache()
    val model = ALS.train(data,5,10)
    (model.userFeatures,model.productFeatures)
  }
}
