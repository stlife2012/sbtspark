package com.stlife.jqxx.c9

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.{SparseVector => SV}
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.regression.LabeledPoint

object TFIDFApp{
  def main(args:Array[String]): Unit ={
    val sc = new SparkContext("local[1]","c9")
    val rdd = sc.wholeTextFiles("data/20news-bydate-train/*")//rec.sport.hockey
    val newsgroups = rdd.map { case (file, text) => file.split("/").takeRight(2).head }
    val countByGroup = newsgroups.map(n => (n, 1)).reduceByKey(_ + _).collect.sortBy(-_._2).mkString("\n")
    println(countByGroup)

    // Tokenizing the text data
    val text = rdd.map { case (file, text) => text }
//    val whiteSpaceSplit = text.flatMap(t => t.split(" ").map(_.toLowerCase))
//    println(whiteSpaceSplit.distinct.count)
    // 402978
    // inspect a look at a sample of tokens - note we set the random seed to get the same results each time
//    println(whiteSpaceSplit.sample(true, 0.3, 42).take(100).mkString(","))

    // split text on any non-word tokens
    val nonWordSplit = text.flatMap(t => t.split("""\W+""").map(_.toLowerCase))
//    println(nonWordSplit.distinct.count)

    // inspect a look at a sample of tokens
//    println(nonWordSplit.distinct.sample(true, 0.3, 42).take(100).mkString(","))

    // filter out numbers
    val regex = """[^0-9]*""".r
    val filterNumbers = nonWordSplit.filter(token => regex.pattern.matcher(token).matches)
//    println(filterNumbers.distinct.count)
    // 84912
//    println(filterNumbers.distinct.sample(true, 0.3, 42).take(100).mkString(","))

    // examine potential stopwords
    val tokenCounts = filterNumbers.map(t => (t, 1)).reduceByKey(_ + _)
//    val oreringDesc = Ordering.by[(String, Int), Int](_._2)
//    println(tokenCounts.top(20)(oreringDesc).mkString("\n"))

    // filter out stopwords
    val stopwords = Set(
      "the","a","an","of","or","in","for","by","on","but", "is", "not", "with", "as", "was", "if",
      "they", "are", "this", "and", "it", "have", "from", "at", "my", "be", "that", "to"
    )
//    val tokenCountsFilteredStopwords = tokenCounts.filter { case (k, v) => !stopwords.contains(k) }
//    println(tokenCountsFilteredStopwords.top(20)(oreringDesc).mkString("\n"))

    // filter out tokens less than 2 characters
//    val tokenCountsFilteredSize = tokenCountsFilteredStopwords.filter { case (k, v) => k.size >= 2 }
//    println(tokenCountsFilteredSize.top(20)(oreringDesc).mkString("\n"))

    // examine tokens with least occurrence
//    val oreringAsc = Ordering.by[(String, Int), Int](-_._2)
//    println(tokenCountsFilteredSize.top(20)(oreringAsc).mkString("\n"))

    // filter out rare tokens with total occurence < 2
    val rareTokens = tokenCounts.filter{ case (k, v) => v < 2 }.map { case (k, v) => k }.collect.toSet
//    val tokenCountsFilteredAll = tokenCountsFilteredSize.filter { case (k, v) => !rareTokens.contains(k) }
//    println(tokenCountsFilteredAll.top(20)(oreringAsc).mkString("\n"))

    def tokenize(line: String): Seq[String] = {
      line.split("""\W+""")
        .map(_.toLowerCase)
        .filter(token => regex.pattern.matcher(token).matches)
        .filterNot(token => stopwords.contains(token))
        .filterNot(token => rareTokens.contains(token))
        .filter(token => token.size >= 2)
        .toSeq
    }

    // check that our tokenizer achieves the same result as all the steps above
    println(text.flatMap(doc => tokenize(doc)).distinct.count)

    // tokenize each document
    val tokens = text.map(doc => tokenize(doc))
    tokens.saveAsObjectFile("out/tokens/token1")
    println(tokens.first.take(20))

    // === train TF-IDF model === //
    val dim = math.pow(2, 18).toInt
    val hashingTF = new HashingTF(dim)
    val tf = hashingTF.transform(tokens)
    tf.cache
    val v = tf.first.asInstanceOf[SV]
    println(v.size)

    val idf = new IDF().fit(tf)
    val tfidf = idf.transform(tf)
    val v2 = tfidf.first.asInstanceOf[SV]
    println(v2.values.size)

    // test out a few common words
    val common = sc.parallelize(Seq(Seq("you", "do", "we")))
    val tfCommon = hashingTF.transform(common)
    val tfidfCommon = idf.transform(tfCommon)
    val commonVector = tfidfCommon.first.asInstanceOf[SV]
    println(commonVector.values.toSeq)

    // test out a few uncommon words
    val uncommon = sc.parallelize(Seq(Seq("telescope", "legislation", "investment")))
    val tfUncommon = hashingTF.transform(uncommon)
    val tfidfUncommon = idf.transform(tfUncommon)
    val uncommonVector = tfidfUncommon.first.asInstanceOf[SV]
    println(uncommonVector.values.toSeq)

    // === document similarity === //
    val hockeyText = rdd.filter { case (file, text) => file.contains("guns") }
    // note that the 'transform' method used below is the one which works on a single document
    // in the form of a Seq[String], rather than the version which works on an RDD of documents
    val hockeyTF = hockeyText.mapValues(doc => hashingTF.transform(tokenize(doc)))
    val hockeyTfIdf = idf.transform(hockeyTF.map(_._2))

    // compute cosine similarity using Breeze
    import breeze.linalg._
    val hockey1 = hockeyTfIdf.sample(true, 0.1, 42).first.asInstanceOf[SV]
    val breeze1 = new SparseVector(hockey1.indices, hockey1.values, hockey1.size)
    val hockey2 = hockeyTfIdf.sample(true, 0.1, 43).first.asInstanceOf[SV]
    val breeze2 = new SparseVector(hockey2.indices, hockey2.values, hockey2.size)
    val cosineSim = breeze1.dot(breeze2) / (norm(breeze1) * norm(breeze2))
    println(f"两个同类文章的相似度：${cosineSim}")

    // compare to comp.graphics topic
    val graphicsText = rdd.filter { case (file, text) => file.contains("hockey") }
    val graphicsTF = graphicsText.mapValues(doc => hashingTF.transform(tokenize(doc)))
    val graphicsTfIdf = idf.transform(graphicsTF.map(_._2))
    val graphics = graphicsTfIdf.sample(true, 0.1, 42).first.asInstanceOf[SV]
    val breezeGraphics = new SparseVector(graphics.indices, graphics.values, graphics.size)
    val cosineSim2 = breeze1.dot(breezeGraphics) / (norm(breeze1) * norm(breezeGraphics))
    println(f"两个不同类文章的相似度：${cosineSim2}")

    // compare to sport.baseball topic
    val baseballText = rdd.filter { case (file, text) => file.contains("misc") }
    val baseballTF = baseballText.mapValues(doc => hashingTF.transform(tokenize(doc)))
    val baseballTfIdf = idf.transform(baseballTF.map(_._2))
    val baseball = baseballTfIdf.sample(true, 0.1, 42).first.asInstanceOf[SV]
    val breezeBaseball = new SparseVector(baseball.indices, baseball.values, baseball.size)
    val cosineSim3 = breeze1.dot(breezeBaseball) / (norm(breeze1) * norm(breezeBaseball))
    println(cosineSim3)

    val newsgroupsMap = newsgroups.distinct.collect().zipWithIndex.toMap
    val zipped = newsgroups.zip(tfidf)
    val train = zipped.map { case (topic, vector) => LabeledPoint(newsgroupsMap(topic), vector) }
    train.cache
    val model = NaiveBayes.train(train, lambda = 0.1)

    val testPath = "data/20news-bydate-train/*"
    val testRDD = sc.wholeTextFiles(testPath)
    val testLabels = testRDD.map { case (file, text) =>
      val topic = file.split("/").takeRight(2).head
      newsgroupsMap(topic)
    }
    val testTf = testRDD.map { case (file, text) => hashingTF.transform(tokenize(text)) }
    val testTfIdf = idf.transform(testTf)
    val zippedTest = testLabels.zip(testTfIdf)
    val test = zippedTest.map { case (topic, vector) => LabeledPoint(topic, vector) }

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    println("accuracy:" + accuracy)
    // 0.7915560276155071
    val metrics = new MulticlassMetrics(predictionAndLabel)
    println("weightedFMeasure:" + metrics.weightedFMeasure)
    // 0.7810675969031116

    // test on raw token features
    val rawTokens = rdd.map { case (file, text) => text.split(" ") }
    val rawTF = rawTokens.map(doc => hashingTF.transform(doc))
    val rawTrain = newsgroups.zip(rawTF).map { case (topic, vector) => LabeledPoint(newsgroupsMap(topic), vector) }
    val rawModel = NaiveBayes.train(rawTrain, lambda = 0.1)
    val rawTestTF = testRDD.map { case (file, text) => hashingTF.transform(text.split(" ")) }
    val rawZippedTest = testLabels.zip(rawTestTF)
    val rawTest = rawZippedTest.map { case (topic, vector) => LabeledPoint(topic, vector) }
    val rawPredictionAndLabel = rawTest.map(p => (rawModel.predict(p.features), p.label))
    val rawAccuracy = 1.0 * rawPredictionAndLabel.filter(x => x._1 == x._2).count() / rawTest.count()
    println("rawAccuracy:" + rawAccuracy)
    // 0.7661975570897503
    val rawMetrics = new MulticlassMetrics(rawPredictionAndLabel)
    println("weightedFMeasure:" + rawMetrics.weightedFMeasure)
    // 0.7628947184990661

    import org.apache.spark.mllib.feature.Word2Vec
    val word2vec = new Word2Vec()
    word2vec.setSeed(42) // we do this to generate the same results each time
    val word2vecModel = word2vec.fit(tokens)
    word2vecModel.findSynonyms("hockey", 20).foreach(println)
    word2vecModel.findSynonyms("legislation", 20).foreach(println)
  }

}
