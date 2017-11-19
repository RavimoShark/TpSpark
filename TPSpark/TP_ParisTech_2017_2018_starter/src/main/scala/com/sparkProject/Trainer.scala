package com.sparkProject
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()
    import spark.implicits._
    /** *****************************************************************************
      *
      * TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    /** CHARGER LE DATASET **/


    val DataPrepClean = spark.read.load("/home/joseph/Dropbox/DeepLearning/Programmation/Spark/TPSpark/funding-successful-projects-on-kickstarter/DataPreprocessed/TrainDataPrep")
    DataPrepClean.printSchema()

    /** TF-IDF **/

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //val TextData = tokenizer.transform(DataPrepClean)
    //DataPrepClean.createOrReplaceTempView("DataClean")
    //TextData.select("tokens").show(false)

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    //val DataPrepCleanRm = remover.transform(DataPrepClean)
    //DataPrepCleanRm.select("filtered").show(false)


    val cvModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("textcount")


    //val dataPrepCleanRmVec = cvModel.transform(DataPrepCleanRm)

    val idf = new IDF().setInputCol("textcount").setOutputCol("tfidf")
    //val idfModel = idf.fit(dataPrepCleanRmVec)

    //val rescaledData = idfModel.transform(dataPrepCleanRmVec)
    //rescaledData.select("tfidf","texcount").show()

    val indexercountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val indexercurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    // val dataindexed = indexercountry.fit(rescaledData).transform(rescaledData)
    //dataindexed.select("country_indexed")show()

    //val dataindexed_2 = indexercurrency.fit(dataindexed).transform(dataindexed)
    //dataindexed_2.select("currency_indexed").show()

    /** VECTOR ASSEMBLER **/

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")

    //val outputdata = assembler.transform(dataindexed_2)
    //println("\"tfidf\", \"days_campaign\", \"hours_prepa\", \"goal\", \"country_indexed\", \"currency_indexed\"' to vector column 'features'")
    //outputdata.select("features").show()


    /** MODEL **/

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions") //raw_predictions
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** PIPELINE **/
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel, idf, indexercountry,
        indexercurrency, assembler, lr))


    val Array(training, test) = DataPrepClean.randomSplit(Array[Double](0.9, 0.1), seed = 1)


    /** TRAINING AND GRID-SEARCH **/
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, (8 to 2 by -2).map(el => math.pow(10, -1 * el)))
      .addGrid(cvModel.minDF, (55.0 to 95.0 by 20.0))
      .build()

    val evaluatorF1 = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val BCE = new BinaryClassificationEvaluator()
      .setLabelCol("final_status")
      .setRawPredictionCol("raw_predictions")
      .setMetricName("areaUnderPR")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(BCE)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4) // Use 3+ in practice

    // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluatorF1)
      .setEstimatorParamMaps(paramGrid)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)




    val bestModel = trainValidationSplit.fit(training)
//    val bestModel_1 = cv.fit(training)


    val df_WithPredictions = bestModel.transform(test)
//    val df_WithPredictions_1 = bestModel_1.transform(test)

    println("TrainValidation:")
    df_WithPredictions.groupBy("final_status","predictions").count.show()

//    println("CrossValidation:")
//    df_WithPredictions_1.groupBy("final_status","predictions").count.show()


    // Displaying the parameters found via grid search
    val bestPipelineModel = bestModel.bestModel.asInstanceOf[PipelineModel]
    val bestPipelineModel_1 = bestModel.bestModel.asInstanceOf[PipelineModel]
    val stages = bestPipelineModel.stages
    val stages_1 = bestPipelineModel.stages

    println("Best parameters found on grid search for Train Val :")
    val hashingStage = stages(2).asInstanceOf[CountVectorizerModel]
    println("\tminDF = " + hashingStage.getMinDF)
    val lrStage = stages(stages.length - 1).asInstanceOf[LogisticRegressionModel]
    println("\tregParam = " + lrStage.getRegParam)
    //f1 score on test set
    val f1 = evaluatorF1.evaluate(df_WithPredictions)
    println("Model F1: " + f1)

    /*println("Best parameters found on grid search for Cross Val :")
    val hashingStage_1 = stages_1(2).asInstanceOf[CountVectorizerModel]
    println("\tminDF =_1 " + hashingStage_1.getMinDF)
    val lrStage_1 = stages_1(stages_1.length - 1).asInstanceOf[LogisticRegressionModel]
    println("\tregParam = " + lrStage_1.getRegParam)
    //f1 score on test set
    val f1_1 = evaluatorF1.evaluate(df_WithPredictions_1)
    println("Model F1: " + f1_1)
*/
  }
}
