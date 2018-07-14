package com.stlife.jqxx.c8

import java.awt.image.BufferedImage

import javax.imageio.ImageIO
import java.io.File

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.DenseMatrix
import breeze.linalg.csvwrite

object DataLoad {
  val baseDir = "G:/data/lfw"
  val aePath = baseDir + "/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
  val outPath = "G:\\data\\out\\lwf\\"

  def main(args:Array[String]): Unit ={
    val sc = new SparkContext("local[2]","c8")
    val rdd = sc.wholeTextFiles("G:/data/lfw/*")
    val files = rdd.map { case (fileName, content) => fileName.replace("file:", "") }
    files.top(5).foreach(println)
    val fp = files.first()
//    println(fp)

//    val fp1 = "G:\\data\\prj\\T02-GYZH-K0001\\img002830.jpg"
//    val aeImage = loadImageFromFile(fp1)
//    val w = 600
//    val a = 1.408
//    val h = w * a
//    val grayImage = processImage(aeImage, w, h.toInt)
//    print(grayImage)
//    ImageIO.write(grayImage, "jpg", new File(outPath + fp1.substring(fp1.lastIndexOf("\\") + 1)))

    //将一个图片转化为数组
//    val data = extractPixels(fp,100,100)
//    println(data.mkString(" "))
//    println(data.length)

    //将一个图片转化为一个向量
    val pixels = files.map(f => extractPixels(f, 50, 50))
    val vectors = pixels.map(p => Vectors.dense(p))
    // the setName method createa a human-readable name that is displayed in the Spark Web UI
    vectors.setName("image-vectors")
    // remember to cache the vectors to speed up computation
    vectors.cache

    //正则化数据
    val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)
    val scaledVectors = vectors.map(v => scaler.transform(v))

//    scaledVectors.saveAsTextFile("out/vectors/vec1")
    //数据降维
    val matrix = new RowMatrix(scaledVectors)
    val K = 10
    val pc = matrix.computePrincipalComponents(K)
    val rows = pc.numRows
    val cols = pc.numCols
    println(rows, cols)

    //保存主成分数据
//    import breeze.linalg.DenseMatrix
//    val pcBreeze = new DenseMatrix(rows, cols, pc.toArray)
//    import breeze.linalg.csvwrite
//    import java.io.File
//    csvwrite(new File("out/vectors/pc.csv"), pcBreeze)

    val projected = matrix.multiply(pc)//原数据向量X与主成分 2x2500 2500x10 = 2x10
    println(projected.numRows, projected.numCols)
    println(projected.rows.take(5).mkString("\n"))
    projected.rows.saveAsTextFile("out/vectors/data.csv")
  }
  //将一个图片进行灰度及大小尺寸改变处理
  def extractPixels(path: String, width: Int, height: Int): Array[Double] = {
    val raw = loadImageFromFile(path)
    val processed = processImage(raw, width, height)
    getPixelsFromImage(processed)
  }

  def getPixelsFromImage(image: BufferedImage): Array[Double] = {
    val width = image.getWidth
    val height = image.getHeight
    val pixels = Array.ofDim[Double](width * height)
    image.getData.getPixels(0, 0, width, height, pixels)
    // pixels.map(p => p / 255.0) 		// optionally scale to [0, 1] domain
  }

  def loadImageFromFile(path: String): BufferedImage = {
    import javax.imageio.ImageIO
    import java.io.File
    ImageIO.read(new File(path))

  }

  def processImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
    val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    val g = bwImage.getGraphics()
    g.drawImage(image, 0, 0, width, height, null)
    g.dispose()
    bwImage
  }

}
