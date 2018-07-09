package com.stlife.jqxx.c9

import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.rdd.RDD

import scala.collection.mutable

object WordVecApp {
  def main(arg:Array[String]): Unit ={
    val sc = new SparkContext("local[1]","wordvec")
//    val tokens = sc.textFile("out/tokens/token").map(line=>line.split(",").toSeq)
    val tokens:RDD[mutable.Seq[String]] = sc.objectFile("out/tokens/token1")
//    tokens.take(10).foreach(line=>println(line))
    val word2vec = new Word2Vec()
    word2vec.setSeed(42) // we do this to generate the same results each time
    word2vec.setNumIterations(10)
    val word2vecModel = word2vec.fit(tokens)

    val local = word2vecModel.getVectors.map{
      case (word, vector) =>Seq(word, vector.mkString(" ")).mkString(":")
    }.toArray
//    sc.parallelize(local).saveAsTextFile("out/word2vec/vec")

//    word2vecModel.findSynonyms("hockey", 20).foreach(println)
    word2vecModel.findSynonyms("legislation", 20).foreach(println)
//    println()
//    word2vecModel.getVectors.get("looks").foreach(line=>println(line))

  }
}
