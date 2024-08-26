#Comparative Analysis of Linear and Non-Linear Models for Regression Tasks
Create linear and non-linear model to solve a regression problem. Evalute the model performance computing the mean squared error (MSE) on a test set.

Task 1
We need to use the following family of models to fit the data: 
f(x, theta) = theta_0 + theta_1 * x_1 + theta_2 * x_2 + theta_3 * sin(x_2) + theta_4 * x_1 * x_2

We start the code by splitting data into a training and a test set, where the training set has a size of 85% of the total data set. In particular, we can see the shape of each set: (1700, 2) considering the training set of x, as it is a bi-dimensional vector, and (300,2) for the test set. The shape of y is the same but unidimensional. Then we build a matrix X to represent the input in a compact form. In particular, there is a column of ones to take into account the intercept, and other four vectors representing the features of the family of models (x_1, x_2, sin(x_2), x_1 * x_2). This matrix is the input that we use to solve the linear system with the function np. linalg.solve() to find the optimal parameters (theta). Another way to find them is by using the LinearRegression model from the sklearn library. In both cases, we find the same estimates of the parameters, which are:

f (x, θ) = 1.2731375 – 0.0377395 · x1 – 0.56711074 · x2 + 0.42078959· sin(x2) + 0.03463118 · x1 · x2

Then, we can evaluate the performance of this model by computing the mean squared error using the test set. 
The mean squared error is defined as:
MSE=1/n ∑▒〖(yi-(yi) ̂)^2 〗
Where yi is the true value and (yi) ̂ is the predicted value of the response variable. We can find the predicted value using the Linear_function that fits data to the family of models we have to consider.  In this case, the MSE of the test set is 0.768 (0.71 on the whole dataset), which is slightly higher than that computed using the training set, 0.70. This is due to the fact that the training set is already used to fit the data, and therefore its evaluation is biased. While the test set is used only to assess the performance of the model.



Task 2
Now we consider a non-linear model to solve the same regression problem. In particular, we will build a model based on neural networks that use the stochastic gradient descent (SGD) optimizer. First of all, we set a random seed (42) to ensure the reproducibility of the results. Then we create a Keras Sequential model, which is simply a linear stack of layers. In this case, there is one input layer followed by three hidden layers with 10 neurons each and the same activation function, which is ReLu. The process ends with the output layers composed of one neuron and using the Linear activation function, given that this is a regression task. Then we configure the model for training using the function ‘model_compile()’, in which we set the objective function to minimize during the training that, in this case, is the mean squared error. As anticipated, we choose the stochastic gradient descent as the optimizer, while the metric to evaluate the performance is still the MSE. Before fitting the model, we also define an early stopping to prevent overfitting. In this case, it will stop the training process if the validation loss does not improve for 20 consecutive epochs, which is the number of iterations of the model during the process. Now, we can fit the model using 200 epochs and a batch size of 32. Moreover, we use 15% of training data as the validation set.
If we compute the MSE with the test set we obtain a value of 0.12, also on the whole dataset, which is significantly lower than before, when we used the linear model. Therefore, this model seems to be much more accurate than the first one, since the error is lower, its performance is better.

Task 3

In the last task, we need to provide a model with an MSE lower than 0.022. Also in this case, we can build a neural network to fit the data, given that we have seen it is better than the LinearRegression model, changing a little the structure given that we use different input data and we have a target performance to achieve. This time, we choose to apply the Adam optimizer and increase the number of neurons to 14 for each layer. We maintain the same number of hidden layers and the same activation functions. However, we increase the number of epochs from 200 to 300, in order to obtain greater accuracy. Finally, we can evaluate the model by computing the mean squared error and we obtain a MSE of 0.0209 on the test set and of 0.018 on the whole dataset.
