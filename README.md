This repo consists of two directories.

One containing a basic example of a pipeline used to make predictions

And another with more advanced and tuned pipeline model

The output of the advanced script has a r squared value of 0.9, meaning that the model can explain 90% of the variance of the depenedent variable (housing price) based on the input variables

Additionally, there is a visual of the cross validation scores by different alpha values used to hyper tune an elastic search function

The hypertuning is a process in machine learning wherein some variable is used to optimize a selected models results

In this example a gridsearch over possible alpha and L1 variables are tried, and different cross validation scores are generated

Some L1 and alpha values result in better model performance (up until a point), in the "Grid Search Scores.png" there is an upper bound reached at different L1 and alpha values
