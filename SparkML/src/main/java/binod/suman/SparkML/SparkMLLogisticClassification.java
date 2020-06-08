package binod.suman.SparkML;



import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.Stream;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;

import scala.Tuple2;

public class SparkMLLogisticClassification {

	public static void main(String[] args) throws IOException {
		
		SparkMLLogisticClassification demo = new SparkMLLogisticClassification();
		JavaSparkContext sc = demo.createSparkContext();
		// Data preparation
		String inputFile = "Master_dataset.csv";
		JavaRDD<LabeledPoint> parsedData = loadDataFromFileAndDataPreparation(sc, inputFile);
		LogisticRegressionModel model = demo.dataSplitAndModelCreationAndAccuracy(parsedData);
		
		/*
		 * //Saving and Retrieval model String modelSavePath =
		 * "model\\logistic-regression"; demo.modelSaving(model, sc, modelSavePath);
		 * model = demo.loadModel(sc, modelSavePath);
		 */
		
		double[] testData = new double[] { 1.0, 1.0, 9839.64, 170136.0, 160296.36, 0.0, 0.0};
		demo.newDataPrediction(model, testData);
		// Close Spark Context
		sc.close();
	}

	public JavaSparkContext createSparkContext() {
		SparkConf conf = new SparkConf().setAppName("Main")
				.setMaster("local[2]")
				.set("spark.executor.memory", "3g")
				.set("spark.driver.memory", "3g");

		JavaSparkContext sc = new JavaSparkContext(conf);
        return sc;
	}

		

	public LogisticRegressionModel dataSplitAndModelCreationAndAccuracy(JavaRDD<LabeledPoint> parsedData) {
		// 5. Data Splitting into 80% Training and 20% Test Sets
		JavaRDD<LabeledPoint>[] splits = parsedData.randomSplit(new double[] { 0.8, 0.2 }, 11L);
		JavaRDD<LabeledPoint> trainingData = splits[0].cache();
		JavaRDD<LabeledPoint> testData = splits[1];

		RDD<LabeledPoint> rdd = trainingData.rdd();

		LogisticRegressionModel model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(rdd);
		
		// 6.2. Model Evaluation
		JavaPairRDD<Object, Object> predictionAndLabels = testData.mapToPair(p -> 
			new Tuple2<>(model.predict(p.features()), p.label()));
		
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		double accuracy = metrics.accuracy();
		System.out.println("Model Accuracy on Test Data: " + accuracy);
		return model;
	}
	
	public void modelSaving(LogisticRegressionModel model, JavaSparkContext sc, String modelSavePath) {
	     model.save(sc.sc(), modelSavePath);       
	}
	
	public LogisticRegressionModel loadModel(JavaSparkContext sc, String modelSavePath) {
		 LogisticRegressionModel model = LogisticRegressionModel.load(sc.sc(), modelSavePath);
		 return model;
	}
	
	public int newDataPrediction(LogisticRegressionModel model, double[] testData) {

		Vector newData = Vectors.dense(testData);
		double prediction = model.predict(newData);
		System.out.println("Prediction label for new data given : " + prediction);
        return (int)prediction;
	}

	protected static JavaRDD<LabeledPoint> loadDataFromFileAndDataPreparation(JavaSparkContext sc, String inputFile) throws IOException {
		File file = new File(inputFile);
		JavaRDD<String> data = sc.textFile(file.getPath());

		// Removing the header from CSV file
		String header  = data.first(); 
		data = data.filter(line ->  !line.equals(header) );

		return data.
				map(line -> {
					//System.out.println(line);
					line = line.replace("PAYMENT", "1")
							.replace("TRANSFER", "2")
							.replace("CASH_OUT", "3")
							.replace("DEBIT", "4")
							.replace("CASH_IN", "5");
					String[] split = line.split(",");

					double[] featureValues = Stream.of(split)
							.mapToDouble(e -> Double.parseDouble(e)).toArray();

					if (featureValues.length > 7) {
						double label = featureValues[7];
						featureValues = Arrays.copyOfRange(featureValues, 0, 7);
						return new LabeledPoint(label, Vectors.dense(featureValues));
					}
					return null;
				}).cache();
	}

	

}
