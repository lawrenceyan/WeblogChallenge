package org.deeplearning4j.examples.dataexamples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SessionUniqueVisits {

    private static Logger log = LoggerFactory.getLogger(SessionUniqueVisits.class);

    public static void main(String[] args) throws  Exception {

        // First: get the dataset using the record reader. CSVRecordReader handles loading/parsing. 
        // Dataset consists of two fields, session ip and unique url visits, from output of running sessionize.pig script 
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("data/session_ip_and_unique_url_visits").getFile()));

        // Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 2;     // 2 values in each row of the file CSV: 1 input feature (Session IP) followed by an integer label (Unique URL Visits) index. 
        int batchSize = 404391;    // number of training/testing data variables to be loaded into dataset

        // Normally would distribute processes, but for purposes of challenge will solely be run locally. 
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex);
        DataSet allData = iterator.next();
        allData.shuffle(); // randomize data for splitting between training and testing
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  // Use 65% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        final int numInputs = 4;
        int outputNum = 3;
        int iterations = 1000;
        long seed = 6;

        // Build model
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .activation(Activation.RELU) // Rectified Linear Unit: defined f(x)=max(0,x)
            .weightInit(WeightInit.XAVIER) // Xavier Initialization scales initialization based on the number of input/output neurons
            .learningRate(0.1)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) // Minimizing negative log likelihood is equivalent to finding maximum likelihood estimation
                .activation(Activation.RELU)
                .nIn(3).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();

        // Run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        model.fit(trainingData);

        // Evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());
    }