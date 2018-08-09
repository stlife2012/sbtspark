package com.stlife.hdfs.prj

import java.net.URI
import java.text.SimpleDateFormat
import java.util.Date

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.compress.BZip2Codec
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object KProcess {
  def main(args: Array[String]): Unit = {
      if(args != null && args.length > 0){
        println("args0:" + args(0))
        println("args1:" + args(1))
        process(args(0),args(1))
      }else{
        process("/data/prj/test","/data/prj/model/kmeans_test4")///data/prj/test /data/prj/tar
      }
  }

  def process(path:String,modelPath:String): Unit ={
    val hdfs: String = "hdfs://master2:9000"
    val outDir = "/data/prj/out"
    val sc = setContext()
    var fp = "/data/prj/tar/gy_data.txt.bz2" //gywc_data.txt.bz2  hdfs://master2:9000/data/prj/tar/gy_data.txt.bz2
    if(path.length > 1){
      fp = path
    }
    val files = fileList(hdfs,path)
    println(f"file list:${files.mkString(",")}")
    val data = sc.textFile(files.mkString(","),100)
    println(f"file dim:${data.first().size}")

    val acc = sc.longAccumulator("acc")
    val fVecData = data.mapPartitions(part=> {
      val content = part.map(f=>f.split(","))
      content.map(f=>{
        val p = f(1).split("\\|").filter(_.length > 0).map(_.toDouble)
        (f(0),f(0).hashCode.toLong,Vectors.dense(p))
      })
    }
    )
    fVecData.map(f=>f._2).repartition(1).saveAsTextFile(hdfs + "/data/prj/model/" + getTimeStr())
    transfer(fVecData,hdfs,outDir,modelPath,sc)
    sc.stop()
  }

  def transfer(vectors:RDD[(String,Long,Vector)],hdfs:String,outDir:String,modelPath:String,sc:SparkContext): Unit ={
    vectors.setName("image-vectors")
    val vec = vectors.mapPartitions(part => part.map(f=>f._3))//获取文件数据
    //正则化数据
    val scaler = new StandardScaler(withMean = true, withStd = false).fit(vec)
    val scaledVectors = vec.map(v => scaler.transform(v))

    //转化文件与数据的对应
    val idxRow = vectors.map(f=>IndexedRow(f._2,f._3))
    val mat: IndexedRowMatrix = new IndexedRowMatrix(idxRow)
    println("mat:" + mat.numRows() + " " + mat.numCols() + " ")

    //数据降维
    val matrix = new RowMatrix(scaledVectors)
    val K = 300
    val svd = matrix.computeSVD(K,false)
    println(s"S dimension: (${svd.s.size}, )")

    val nidxMat = mat.multiply(svd.V)//数据降维 7x2500 2500x2 = 7x2
    val fIdx = vectors.map(f=>(f._2,f._1))//获取文件数据
    val idxVec = nidxMat.rows.map(f=>(f.index,f.vector))
    val fIdxVec = fIdx.zip(idxVec).map(f=>(f._1._1,f._1._2,f._2._2))//path id vec
    println(fIdxVec.first())

    println("nidxMat:" + nidxMat.numRows, nidxMat.numCols)
//    fIdxVec.saveAsTextFile(hdfs + outDir + "/dim_10" + getTimeStr(),classOf[BZip2Codec])
    fIdxVec.saveAsTextFile(hdfs + outDir + "/dim_10" + getTimeStr())

    //模型计算
    val dataV = nidxMat.rows.map(f=>f.vector)
    val model = KMeans.train(dataV,100,50)
    model.save(sc,hdfs + modelPath)
  }

  def setContext(): SparkContext ={
    val master: String = "spark://master2:7077"
//    val master: String = "local[*]"
    val conf = new SparkConf().setAppName("DataProcess").setMaster(master)
    conf.set("spark.memory.fraction","0.9")
    val sc = new SparkContext(conf)
    println(s"spark.memory.fraction=>${sc.getConf.get("spark.memory.fraction")}")
    sc
  }

  def fileList(hdfs:String,dir:String): List[Path] ={
    val conf: Configuration = new Configuration()
    val fs: FileSystem = FileSystem.get(new URI(hdfs), conf)
    val filesStatus = fs.listStatus(new Path(hdfs + dir))
    val files = filesStatus.map(f=>f.getPath).toList
    files
  }
  def getTimeStr(): String ={
    val now: Date = new Date()
    val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyyMMddHHmmss")
    val date = dateFormat.format(now)
    date.toString
  }

  def readFile(data:RDD[String],sc:SparkContext,idx:Int,files:List[Path]): RDD[String] ={
    if(idx < files.size){
      val curData = sc.textFile(files(idx).toString,100)
      val udata = data.union(curData)
      readFile(udata,sc,idx+1,files)
    }else {
      data
    }
  }
}
