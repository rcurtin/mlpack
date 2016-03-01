/**
 * @file hoeffding_forest_main.cpp
 * @author Ryan Curtin
 *
 * A command-line executable that can build a streaming decision tree.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_forest.hpp>
#include <mlpack/methods/hoeffding_trees/binary_numeric_split.hpp>
#include <mlpack/methods/hoeffding_trees/information_gain.hpp>
#include <mlpack/methods/hoeffding_trees/single_random_dimension_split.hpp>
#include <mlpack/methods/hoeffding_trees/multiple_random_dimension_split.hpp>
#include <queue>

using namespace std;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;

PROGRAM_INFO("Hoeffding random forests",
    "This program implements Hoeffding random forests, a random forest "
    "algorithm that uses a form of streaming decision tree"
    " suited best for large (or streaming) datasets.  This program supports "
    "both categorical and numeric data stored in the ARFF format.  Given an "
    "input dataset, this program is able to train the tree with numerous "
    "training options, and save the model to a file.  The program is also able "
    "to use a trained model or a model from file in order to predict classes "
    "for a given test set."
    "\n\n"
    "The training file and associated labels are specified with the "
    "--training_file and --labels_file options, respectively.  The training "
    "file must be in ARFF format.  The training may be performed in batch mode "
    "(like a typical decision tree algorithm) by specifying the --batch_mode "
    "option, but this may not be the best option for large datasets."
    "\n\n"
    "When a model is trained, it may be saved to a file with the "
    "--output_model_file (-M) option.  A model may be loaded from file for "
    "further training or testing with the --input_model_file (-m) option."
    "\n\n"
    "A test file may be specified with the --test_file (-T) option, and if "
    "performance numbers are desired for that test set, labels may be specified"
    " with the --test_labels_file (-L) option.  Predictions for each test point"
    " will be stored in the file specified by --predictions_file (-p) and "
    "probabilities for each predictions will be stored in the file specified by"
    " the --probabilities_file (-P) option.");

PARAM_STRING("training_file", "Training dataset file.", "t", "");
PARAM_STRING("labels_file", "Labels for training dataset.", "l", "");

PARAM_DOUBLE("confidence", "Confidence before splitting (between 0 and 1).",
    "c", 0.95);
PARAM_INT("max_samples", "Maximum number of samples before splitting.", "A",
    5000);
PARAM_INT("min_samples", "Minimum number of samples before splitting.", "I",
    100);
PARAM_INT("forest_size", "Number of trees in the forest.", "f", 5);

PARAM_STRING("input_model_file", "File to load trained tree from.", "m", "");
PARAM_STRING("output_model_file", "File to save trained tree to.", "M", "");

PARAM_STRING("test_file", "File of testing data.", "T", "");
PARAM_STRING("test_labels_file", "Labels of test data.", "L", "");
PARAM_STRING("predictions_file", "File to output label predictions for test "
    "data into.", "p", "");
PARAM_STRING("probabilities_file", "In addition to predicting labels, provide "
    "prediction probabilities in this file.", "P", "");

PARAM_STRING("numeric_split_strategy", "The splitting strategy to use for "
    "numeric features: 'domingos' or 'binary'.", "N", "binary");
PARAM_FLAG("batch_mode", "If true, samples will be considered in batch instead "
    "of as a stream.  This generally results in better trees but at the cost of"
    " memory usage and runtime.", "b");
PARAM_FLAG("info_gain", "If set, information gain is used instead of Gini "
    "impurity for calculating Hoeffding bounds.", "i");
PARAM_INT("passes", "Number of passes to take over the dataset.", "n", 1);

PARAM_INT("bins", "If the 'domingos' split strategy is used, this specifies "
    "the number of bins for each numeric split.", "B", 10);
PARAM_INT("observations_before_binning", "If the 'domingos' split strategy is "
    "used, this specifies the number of samples observed before binning is "
    "performed.", "o", 100);

PARAM_STRING("selection_strategy", "Strategy to use for selecting random "
    "dimensions to split on: 'single' or 'multiple'.", "S", "multiple");
PARAM_INT("dimensions_per_split", "Number of dimensions to use at each split. "
    "If 0, the largest integer less than log2(dimensionality) will be used.",
    "d", 0);

PARAM_INT("seed", "Random seed (if not specified, std::time(NULL) will be "
    "used).", "s", 0);

// Helper function to choose fitness function.
void SelectFitnessFunction(const arma::mat& trainingData,
                           const data::DatasetInfo& info);

// Helper function to choose categorical split strategy.
template<typename FitnessFunction>
void SelectCategoricalSplitType(const arma::mat& trainingData,
                                const data::DatasetInfo& info);

// Helper function to choose numeric split strategy.
template<typename FitnessFunction,
         template<typename> class CategoricalSplitType>
void SelectNumericSplitType(
    const arma::mat& trainingData,
    const data::DatasetInfo& info,
    const CategoricalSplitType<FitnessFunction>& cs =
        CategoricalSplitType<FitnessFunction>(1, 1));

// Helper function to choose split selection strategy.
template<typename FitnessFunction,
         template<typename> class CategoricalSplitType,
         template<typename> class NumericSplitType>
void SelectSplitSelectionStrategy(
    const arma::mat& trainingData,
    const data::DatasetInfo& info,
    const CategoricalSplitType<FitnessFunction>& cs,
    const NumericSplitType<FitnessFunction>& ns =
        NumericSplitType<FitnessFunction>(0));

// Helper function for once we have chosen a tree type.
template<typename TreeType>
void PerformActions(
    const arma::mat& trainingData,
    const data::DatasetInfo& info,
    const typename TreeType::SplitSelectionStrategy& strategy);

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  if (CLI::GetParam<int>("seed") == 0)
    math::RandomSeed(std::time(NULL));
  else
    math::RandomSeed((size_t) CLI::GetParam<int>("seed"));

  // Check input parameters for validity.
  const string trainingFile = CLI::GetParam<string>("training_file");
  const string labelsFile = CLI::GetParam<string>("labels_file");
  const string inputModelFile = CLI::GetParam<string>("input_model_file");
  const string testFile = CLI::GetParam<string>("test_file");
  const string predictionsFile = CLI::GetParam<string>("predictions_file");
  const string probabilitiesFile = CLI::GetParam<string>("probabilities_file");
  const string numericSplitStrategy =
      CLI::GetParam<string>("numeric_split_strategy");
  const string selectionStrategy = CLI::GetParam<string>("selection_strategy");

  if ((!predictionsFile.empty() || !probabilitiesFile.empty()) &&
      testFile.empty())
    Log::Fatal << "--test_file must be specified if --predictions_file or "
        << "--probabilities_file is specified." << endl;

  if (trainingFile.empty() && inputModelFile.empty())
    Log::Fatal << "One of --training_file or --input_model_file must be "
        << "specified!" << endl;

  if (!trainingFile.empty() && labelsFile.empty())
    Log::Fatal << "If --training_file is specified, --labels_file must be "
        << "specified too!" << endl;

  if (trainingFile.empty() && CLI::HasParam("batch_mode"))
    Log::Warn << "--batch_mode (-b) ignored; no training set provided." << endl;

  if (CLI::HasParam("passes") && CLI::HasParam("batch_mode"))
    Log::Warn << "--batch_mode (-b) ignored because --passes was specified."
        << endl;

  if (selectionStrategy != "single" && selectionStrategy != "multiple" &&
      selectionStrategy != "all")
    Log::Fatal << "Invalid selection strategy '" << selectionStrategy << "' "
        << "specified.  Must be 'single', 'multiple', or 'all'." << endl;

  // Now, we have to build the type of tree we're using from the options.
  if (!trainingFile.empty())
  {
    // We have to start with the correct dataset information to build the
    // MultipleRandomDimensionSplit correctly (if necessary), so we have to load
    // the training set now.
    arma::mat trainingSet;
    data::DatasetInfo datasetInfo;
    data::Load(trainingFile, trainingSet, datasetInfo, true);
    for (size_t i = 0; i < trainingSet.n_rows; ++i)
      Log::Info << datasetInfo.NumMappings(i) << " mappings in dimension "
          << i << "." << endl;

    SelectFitnessFunction(trainingSet, datasetInfo);
  }
  else
  {
    // We'll be loading from a model, so we don't need a "real" training set or
    // dataset information.
    arma::mat trainingSet;
    data::DatasetInfo datasetInfo;

    SelectFitnessFunction(trainingSet, datasetInfo);
  }
}

// Helper function to choose fitness function.
void SelectFitnessFunction(const arma::mat& trainingSet,
                           const data::DatasetInfo& info)
{
  if (CLI::HasParam("info_gain"))
    SelectCategoricalSplitType<InformationGain>(trainingSet, info);
  else
    SelectCategoricalSplitType<GiniImpurity>(trainingSet, info);
}

// Helper function to choose categorical split strategy.
template<typename FitnessFunction>
void SelectCategoricalSplitType(const arma::mat& trainingSet,
                                const data::DatasetInfo& info)
{
  // There's only one possibility here, but it could be more later...
  SelectNumericSplitType<FitnessFunction,
      HoeffdingCategoricalSplit>(trainingSet, info,
      HoeffdingCategoricalSplit<FitnessFunction>(1, 1));
}

// Helper function to choose numeric split strategy.
template<typename FitnessFunction,
         template<typename> class CategoricalSplitType>
void SelectNumericSplitType(const arma::mat& trainingSet,
                            const data::DatasetInfo& info,
                            const CategoricalSplitType<FitnessFunction>& cs)
{
  const string numericSplitStrategy =
      CLI::GetParam<string>("numeric_split_strategy");
  if (numericSplitStrategy == "domingos")
  {
    const size_t bins = (size_t) CLI::GetParam<int>("bins");
    const size_t observationsBeforeBinning = (size_t)
        CLI::GetParam<int>("observations_before_binning");
    HoeffdingDoubleNumericSplit<FitnessFunction> ns(0, bins,
        observationsBeforeBinning);
    SelectSplitSelectionStrategy<FitnessFunction, CategoricalSplitType,
        HoeffdingDoubleNumericSplit>(trainingSet, info, cs, ns);
  }
  else if (numericSplitStrategy == "binary")
  {
    SelectSplitSelectionStrategy<FitnessFunction, CategoricalSplitType,
        BinaryDoubleNumericSplit>(trainingSet, info, cs);
  }
}

// Helper function to choose split selection strategy.
template<typename FitnessFunction,
         template<typename> class CategoricalSplitType,
         template<typename> class NumericSplitType>
void SelectSplitSelectionStrategy(
    const arma::mat& trainingSet,
    const data::DatasetInfo& info,
    const CategoricalSplitType<FitnessFunction>& cs,
    const NumericSplitType<FitnessFunction>& ns)
{
  const string selectionStrategy = CLI::GetParam<string>("selection_strategy");
  if (selectionStrategy == "single")
  {
    typedef HoeffdingTree<FitnessFunction, NumericSplitType,
        CategoricalSplitType, SingleRandomDimensionSplit> TreeType;
    SingleRandomDimensionSplit<FitnessFunction, NumericSplitType,
        CategoricalSplitType> split(info, 1, cs, ns);

    PerformActions<TreeType>(trainingSet, info, split);
  }
  else if (selectionStrategy == "multiple")
  {
    typedef HoeffdingTree<FitnessFunction, NumericSplitType,
        CategoricalSplitType, MultipleRandomDimensionSplit> TreeType;

    const size_t dimsPerSplit =
        (size_t) CLI::GetParam<int>("dimensions_per_split");
    MultipleRandomDimensionSplit<FitnessFunction, NumericSplitType,
        CategoricalSplitType> mr(info, 1, cs, ns, dimsPerSplit);

    PerformActions<TreeType>(trainingSet, info, mr);
  }
  else if (selectionStrategy == "all")
  {
    typedef HoeffdingTree<FitnessFunction, NumericSplitType,
        CategoricalSplitType, AllDimensionSplit> TreeType;
    AllDimensionSplit<FitnessFunction, NumericSplitType, CategoricalSplitType>
        split(info, 1, cs, ns);

    PerformActions<TreeType>(trainingSet, info, split);
  }
}

template<typename TreeType>
void PerformActions(const arma::mat& trainingSet,
                    const data::DatasetInfo& datasetInfo,
                    const typename TreeType::SplitSelectionStrategy& strategy)
{
  // Load necessary parameters.
  const string labelsFile = CLI::GetParam<string>("labels_file");
  const double confidence = CLI::GetParam<double>("confidence");
  const size_t maxSamples = (size_t) CLI::GetParam<int>("max_samples");
  const size_t minSamples = (size_t) CLI::GetParam<int>("min_samples");
  const string inputModelFile = CLI::GetParam<string>("input_model_file");
  const string outputModelFile = CLI::GetParam<string>("output_model_file");
  const string testFile = CLI::GetParam<string>("test_file");
  const string predictionsFile = CLI::GetParam<string>("predictions_file");
  const string probabilitiesFile = CLI::GetParam<string>("probabilities_file");
  bool batchTraining = CLI::HasParam("batch_mode");
  const size_t passes = (size_t) CLI::GetParam<int>("passes");
  const size_t forestSize = (size_t) CLI::GetParam<int>("forest_size");
  if (passes > 1)
    batchTraining = false; // We already warned about this earlier.

  HoeffdingForest<TreeType>* forest = NULL;
  if (inputModelFile.empty())
  {
    arma::Col<size_t> labelsIn;
    data::Load(labelsFile, labelsIn, true, false);
    arma::Row<size_t> labels = labelsIn.t();

    // Now create the decision tree.
    Timer::Start("tree_training");

    // Make example TreeType that will capture all of the necessary parameters.
    const size_t numClasses = max(labels) + 1;
    TreeType exampleTree(datasetInfo, numClasses, strategy, confidence,
        maxSamples, 100, minSamples);

    forest = new HoeffdingForest<TreeType>(exampleTree, forestSize, numClasses,
        datasetInfo);

    if (passes > 1)
    {
      Log::Info << "Taking " << passes << " passes over the dataset." << endl;
      for (size_t i = 0; i < passes; ++i)
        forest->Train(trainingSet, labels, false);
    }
    else
    {
      forest->Train(trainingSet, labels, batchTraining);
    }
    Timer::Stop("tree_training");
  }
  else
  {
    forest = new HoeffdingForest<TreeType>(1, 1, datasetInfo);
    data::Load(inputModelFile, "hoeffdingForest", *forest, true);

    if (trainingSet.n_elem != 0)
    {
      arma::Col<size_t> labelsIn;
      data::Load(labelsFile, labelsIn, true, false);
      arma::Row<size_t> labels = labelsIn.t();

      // Now create the decision tree.
      Timer::Start("tree_training");
      if (passes > 1)
      {
        Log::Info << "Taking " << passes << " passes over the dataset." << endl;
        for (size_t i = 0; i < passes; ++i)
          forest->Train(trainingSet, labels, false);
      }
      else
      {
        forest->Train(trainingSet, labels, batchTraining);
      }
      Timer::Stop("tree_training");
    }
  }

  if (trainingSet.n_elem != 0)
  {
    // Get training error.
    arma::Row<size_t> predictions;
    forest->Classify(trainingSet, predictions);

    arma::Col<size_t> labelsIn;
    data::Load(labelsFile, labelsIn, true, false);
    arma::Row<size_t> labels = labelsIn.t();

    size_t correct = 0;
    for (size_t i = 0; i < labels.n_elem; ++i)
      if (labels[i] == predictions[i])
        ++correct;

    Log::Info << correct << " out of " << labels.n_elem << " correct "
        << "on training set (" << double(correct) / double(labels.n_elem) *
        100.0 << ")." << endl;
  }

  // The forest is trained or loaded.  Now do any testing if we need.
  if (!testFile.empty())
  {
    arma::mat testSet;
    data::Load(testFile, testSet, const_cast<data::DatasetInfo&>(datasetInfo),
        true);

    arma::Row<size_t> predictions;
    arma::rowvec probabilities;

    Timer::Start("tree_testing");
    forest->Classify(testSet, predictions, probabilities);
    Timer::Stop("tree_testing");

    if (CLI::HasParam("test_labels_file"))
    {
      string testLabelsFile = CLI::GetParam<string>("test_labels_file");
      arma::Col<size_t> testLabelsIn;
      data::Load(testLabelsFile, testLabelsIn, true, false);
      arma::Row<size_t> testLabels = testLabelsIn.t();

      size_t correct = 0;
      for (size_t i = 0; i < testLabels.n_elem; ++i)
      {
        if (predictions[i] == testLabels[i])
          ++correct;
      }
      Log::Info << correct << " out of " << testLabels.n_elem << " correct "
          << "on test set (" << double(correct) / double(testLabels.n_elem) *
          100.0 << ")." << endl;
    }

    if (!predictionsFile.empty())
      data::Save(predictionsFile, predictions);

    if (!probabilitiesFile.empty())
      data::Save(probabilitiesFile, probabilities);
  }

  // Check the accuracy on the training set.
  if (!outputModelFile.empty())
    data::Save(outputModelFile, "hoeffdingForest", *forest, true);

  // Clean up memory.
  delete forest;
}
