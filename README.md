##First, make sure you have Pig and all other relevant sub-dependencies installed, then run sessionize.pig script; multiple files will be created. 

## Processing & Analytical goals:

1. Sessionized web logs will be stored in 'data/sessions' directory after running script. Each row has the following format (IP, Time Spent, Unique URLs Visited, {(individual log)(individual log)...})  Remember that PIG always stores the actual data in a file called part-r-00000 within created directory.

2. Average session time stored in 'data/average_session_time' directory.

3. Each session in sessionized web logs has number of unique URLs visited stored in the second field. Refer back to 1) for format of sessionized web logs.

4. Sessionized web logs have been sorted from greatest to least based on time spent; i.e. IPs with the longest sessions are the beginning of the file.  

##Once data has been created, make sure you have the DeepLearning4js framework on top of Spark, then run SessionLengthPredictor.java and SessionUniqueVisits.java to train predictive models.

## Additional questions for Machine Learning Engineer (MLE) candidates:
1. Predicted expected load (requests/second) determined based on utilizing Poisson probability distribution as our model. We assume that these 4 statements hold true: a) Occurrences of URL requests are Bernoulli random variables. b) The time period we are analyzing can be divided into many smaller sub-periods, ie hour/minutes/seconds/milliseconds/etc. c) The probability of two or more occurrences of a URL request within some sub-period is negligible. d) The probability of an occurrence is random, amortizing to a constant value within a specified time period. Parameters for our Poisson distribution function stored in 'data/poisson_distribution_parameters'.

2. In order to predict the session length for a given IP, I trained a 3 layer Neural Network. Activation Function used: Rectified Linear Unit (ReLu), Weight Initialization: Xavier Initialization (also called Glorot Initialization), Loss Function: Negative Log Likelihood (when minimized is equivalent to finding maximum likelihood estimation) 

3. Similarly for predicting number of unique url visits for a given IP, I trained another Multilayer Perceptron Model. 3 layers, ReLu activation function, Xavier initialization, Minimizing Negative Log Likelihood for loss function.
