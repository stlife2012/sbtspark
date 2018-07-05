package com.stlife.jqxx.c5

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import org.apache.spark.mllib.optimization.SimpleUpdater
import org.apache.spark.mllib.optimization.SquaredL2Updater

object DataApp {
  def main(arg:Array[String]): Unit ={
//    train()
//    dataStd()
//    other()
    lrModelTrain()
  }

  def other(): Unit ={
//    if(sc == null){
    val sc = new SparkContext("local[1]","c5")
//    }
    val data = sc.textFile("data/noheadertrain.tsv").map(line=>line.split("\\t"))
    val cls = data.map(line=>line(3)).distinct().zipWithIndex().collectAsMap()
    cls.foreach(println)
  }

  def train(): Unit ={//分类模型训练
    val numItator = 20
    val treeDepth = 10
//    val data = dataCls()
    val data = dataStd()
    data.take(5).foreach(println)

    val lrModel = LogisticRegressionWithSGD.train(data,numItator)//0.5147
    val svmModel = SVMWithSGD.train(data,numItator)//0.5147
//    val nbayModel = NaiveBayes.train(data)//0.5804
    val decTreeModel = DecisionTree.train(data,Algo.Classification,Gini,treeDepth)//0.6483
//    val model = decTreeModel
//    val trueNum = data.map(f=>{
//      if(f.label == model.predict(f.features)) 1 else 0
//    }).sum()
//    println("acc：" + (trueNum/data.count()).formatted("%.4f"))
//    println("pred:" + model.predict(data.first().features))
//    println(model.getClass.getSimpleName)
//    println(lrModel.getClass.getSimpleName)

    //评估模型PR ROC
    val metrics = Seq(lrModel,svmModel).map(model=>{
      val predAndTrue = data.map(line=>{
        (model.predict(line.features),line.label)
      })
      val mt = new BinaryClassificationMetrics(predAndTrue)
      (model.getClass.getSimpleName,mt.areaUnderPR(),mt.areaUnderROC())
    })

    val treeMetrics = Seq(decTreeModel).map(model=>{
      val predAndTrue = data.map(line=>{
        val score = model.predict(line.features)
        (if(score > 0.5) 1.0 else 0.0,line.label)
      })
      val mt = new BinaryClassificationMetrics(predAndTrue)
      (model.getClass.getSimpleName,mt.areaUnderPR(),mt.areaUnderROC())
    })
    //打印评估值
    val allMetrics = metrics ++ treeMetrics
    allMetrics.foreach{case (name,pr,roc)=>{
      println(f"名称：${name} PR:${pr * 100}%2.4f%% ROC:${roc * 100.0}%2.4f%%")
    }}
  }

  def dataStd():RDD[LabeledPoint]={//特征数据标准化处理及数据分析验证
    val data = dataCls()
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(data.map(f=>f.features))
    val scaledData = data.map(lp => LabeledPoint(lp.label,scaler.transform(lp.features)))

    val rowMatrix = new RowMatrix(data.map(l=>l.features))
    val columnSummary = rowMatrix.computeColumnSummaryStatistics()
    println(s"均值：${columnSummary.mean}")
    println(s"最小：${columnSummary.min}")
    println(s"最大：${columnSummary.max}")
    println(s"非零项：${columnSummary.numNonzeros}")
    println(s"方差：${columnSummary.variance}")
    println(s"标准化前：${data.first().features}")
    println(s"标准化后：${scaledData.first().features}")
    scaledData.cache()
    scaledData
  }

  def dataCls(): RDD[LabeledPoint] ={//构建分类数据
    val sc = new SparkContext("local[1]","c5")
    val data = sc.textFile("data/noheadertrain.tsv").map(line=>line.split("\\t"))
    //数据从第五开始到最后第二位为特征数据，最后一位为标志数据
    println(data.first().mkString("/"))
    //将数据处理成LabeledPoint格式数据
    val tf = data.map(line=>{
      val record = line.map(_.replaceAll("\"",""))
      val lable = record(record.size - 1).toDouble
      val feature = record.slice(4,record.size-1).map(d=>{
        //如果存在？替换为0, <0替换为0
        if (d == "?"){
          0.0
        }else{
          d.toDouble
        }
      }).map(d=>if(d < 0) 0.0 else d)
      LabeledPoint(lable,Vectors.dense(feature))
    })
//    tf.take(3).foreach(println)
//    tf.map(row=>row.label).distinct().foreach(println)
    tf.cache()
    tf
  }

  def lrModelTrain(): Unit ={
    val data = dataStd()
    //迭代次数参数评估
//    val allMetrics = Seq(5,10,20,30,40).map(step => {
//      val lr = new LogisticRegressionWithSGD()
//      lr.optimizer.setNumIterations(step).setUpdater(new SimpleUpdater)
//      val model = lr.run(data)
//
//      val predAndTrue = data.map(line=>{
//        (model.predict(line.features),line.label)
//      })
//      val metrics = new BinaryClassificationMetrics(predAndTrue)
//      (step,metrics.areaUnderPR(),metrics.areaUnderROC())
//    })
    //步长参数评估
//    val allMetrics = Seq(0.001,0.01,0.1,1,10).map(step => {
//      val lr = new LogisticRegressionWithSGD()
//      lr.optimizer.setNumIterations(30).setUpdater(new SimpleUpdater).setStepSize(step)
//      val model = lr.run(data)
//
//      val predAndTrue = data.map(line=>{
//        (model.predict(line.features),line.label)
//      })
//      val metrics = new BinaryClassificationMetrics(predAndTrue)
//      (step,metrics.areaUnderPR(),metrics.areaUnderROC())
//    })

    //  L2正则化参数评估
    val allMetrics = Seq(0.001,0.01,0.1,1,10).map(step => {
      val lr = new LogisticRegressionWithSGD()
      lr.optimizer.setNumIterations(30).setUpdater(new SquaredL2Updater).setStepSize(step)
      val model = lr.run(data)

      val predAndTrue = data.map(line=>{
        (model.predict(line.features),line.label)
      })
      val metrics = new BinaryClassificationMetrics(predAndTrue)
      (step,metrics.areaUnderPR(),metrics.areaUnderROC())
    })

    allMetrics.foreach{case (step,pr,roc)=>{
      println(f"名称：${step} PR:${pr * 100}%2.4f%% ROC:${roc * 100.0}%2.4f%%")
    }}
  }
}
