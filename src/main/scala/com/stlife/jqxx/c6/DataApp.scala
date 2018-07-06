package com.stlife.jqxx.c6

import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.RegressionMetrics
object DataApp {
  def main(arg:Array[String]): Unit ={
    val data = collectData()
    train(data)
  }

  def train(input:RDD[LabeledPoint]): Unit ={//训练及交叉评估
    var splitData = input.randomSplit(Array(0.8,0.2))
    var trainData = splitData(0)
    var testData = splitData(1)

    val mulMetrics = Seq(5,10,15,30,50).map(step=>{
      val lr = new LinearRegressionWithSGD()
      lr.optimizer.setNumIterations(step)
      val model = lr.run(trainData)
      val predAndTrue = testData.map(line=>{
        (model.predict(line.features),line.label)
      })
      println(f"pred:${model.predict(trainData.first().features)} true:${trainData.first().label}")
      val metrics = new RegressionMetrics(predAndTrue)
      (step,metrics.meanSquaredError,metrics.rootMeanSquaredError)
    })

    mulMetrics.foreach(line=>println(f"Step:${line._1} MSE：${line._2} RMSE:${line._3}%.4f"))
  }

  def collectData():RDD[LabeledPoint] = {//构建数据
    val sc = new SparkContext("local[1]","c6")
    val data = sc.textFile("data/bike/nohhour.csv")
    //1,2011-01-01,1,0,1,0,0,6,0,1,0.24,0.2879,0.81,0,3,13,16
    data.take(2).foreach(println)
    val rdd = data.map(line=>{
      val cell = line.split(",")
      val freature = (cell.slice(2,9) ++ cell.slice(10,12)).map(f=>f.toDouble)//cell.slice(2,cell.size-1)
      val label = cell(cell.size-1).toDouble
      LabeledPoint(label,Vectors.dense(freature))
    })
    val stand = new StandardScaler(withMean = true,withStd = true).fit(rdd.map(line=>line.features))
    val scalerData = rdd.map(lp=>LabeledPoint(lp.label,stand.transform(lp.features)))
    scalerData.cache()
    scalerData
  }
}
