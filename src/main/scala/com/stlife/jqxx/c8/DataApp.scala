package com.stlife.jqxx.c8

import java.io.File

import com.stlife.jqxx.c7.DataApp.collectData
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix, csvwrite}

object DataApp {
  def main(str:Array[String]): Unit ={
//    collectData()
    pcAction()
  }

  def pcAction(): Unit ={//主成分处理
    val feature = collectData()
    val userFeatures = feature._1.map(l=>Vectors.dense(l._2))
    val rowMat = new RowMatrix(userFeatures)

    println("COUNT:" + userFeatures.count() + s" rowmat:${rowMat.numRows()} ${rowMat.numCols()}")
    val k = 5
    val pc = rowMat.computePrincipalComponents(k)// U R V(T) = (943,5)(5,5)(15,5)(T)

    val sdata = rowMat.multiply(pc)
    println(sdata.numCols(),sdata.numRows())

    println(f"Rows:${pc.numRows} Cols:${pc.numCols}")
    val mat = new DenseMatrix(pc.numRows,pc.numCols,pc.toArray)
    csvwrite(new File("data/out/pc.csv"),mat)

    val svd = rowMat.computeSVD(5, computeU = true)
    println(s"U dimension: (${svd.U.numRows}, ${svd.U.numCols})")
    println(s"S dimension: (${svd.s.size}, )")
    println(s"V dimension: (${svd.V.numRows}, ${svd.V.numCols})")

    println(mat(0,::))
    println("-------------------")
    println(svd.V)
  }

  def collectData(): (RDD[(Int, Array[Double])], RDD[(Int, Array[Double])]) ={//通过ALS构建隐形特征因子
    val sc = new SparkContext("local[1]","c7")
    val rdd = sc.textFile("data/ml-100k/u.data")
//    println(rdd.map(l=>{
//      val split = l.split("\t")
//      (split(0).toInt,split(1).toInt,split(2).toDouble)
//    }).distinct().count())
//    println("num:" + rdd.count())
    val data = rdd.map(line=>{
      val split = line.split("\t")
      Rating(split(0).toInt,split(1).toInt,split(2).toDouble)
    })
    data.cache()
    val model = ALS.train(data,15,5)
//    println(model.userFeatures.collect().size)
    (model.userFeatures,model.productFeatures)
  }
}
