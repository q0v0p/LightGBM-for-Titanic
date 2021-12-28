using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace LightGBM_for_Titanic
{
    internal class TitanicLightGBM
    {
        private readonly string _trainDataPath = "Data/train.csv";

        public void Train()
        {
            var mlContext = new MLContext(seed: 0);

            var trainingDataView = mlContext.Data.LoadFromTextFile<Passenger>(_trainDataPath, hasHeader: true, separatorChar: ',');
            var splitData = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

            var options = new LightGbmBinaryTrainer.Options
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                Booster = new GossBooster.Options
                {
                    TopRate = 0.3,
                    OtherRate = 0.2
                }
            };

            string[] categoryFeatureNames = { "Sex", "Embarked" };
            var survivedMap = new Dictionary<string, bool> { { "1", true }, { "0", false } };

            var pipeline = mlContext.Transforms.Conversion.MapValue("Label", survivedMap)
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Sex", inputColumnName: categoryFeatureNames[0]))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Embarked", inputColumnName: categoryFeatureNames[1]))
                .Append(mlContext.Transforms.Concatenate("Features", new[] { "Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked" }))
                .Append(mlContext.BinaryClassification.Trainers.LightGbm(options));


            var model = pipeline.Fit(splitData.TrainSet);

            var transformedTestData = model.Transform(splitData.TestSet);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data
                .CreateEnumerable<TitanicPrediction>(transformedTestData, reuseRowObject: false).ToList();

            // Print 5 predictions.
            foreach (var p in predictions.Take(5))
                Console.WriteLine($"Label: {p.Label}, "
                    + $"Prediction: {p.PredictedLabel}");

            // Evaluate the overall metrics.
            var metrics = mlContext.BinaryClassification
                .Evaluate(transformedTestData);

            PrintMetrics(metrics);
        }

        private static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: " +
                $"{metrics.NegativePrecision:F2}");

            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: " +
                $"{metrics.PositivePrecision:F2}");

            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}\n");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}
