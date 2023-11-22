import numpy as np # Importing the numpy library for numerical computations
import pandas as pd # Importing the pandas library for data manipulation and analysis
import random # Importing the random library for shuffling the data
random.seed(45) # Setting the random seed to 45
import warnings # Importing the warnings library to ignore warning messages
warnings.filterwarnings('ignore')

# Defining function perceptronImpl
def perceptronImpl(train_data, test_data, firstClass, secondClass, thirdClass, l2_reg):
    
    # Splitting the training data into features and target variable
    X_train = train_data[:, :-1]
    y_train_string = train_data[:, -1]

    # Splitting the test data into features and target variable
    X_test = test_data[:, :-1]
    y_test_string = test_data[:, -1]
    
    # If there is no thirdClass, Binary classification Perceptron algorithm is implemented
    if thirdClass == None:
        
        # Define a dictionary to map string labels to their corresponding ids
        target_var_dict = {firstClass: -1, secondClass: 1}
        y_train = np.array([target_var_dict[yi] for yi in y_train_string])

        target_var_dict = {firstClass: -1, secondClass: 1}
        y_test = np.array([target_var_dict[yi] for yi in y_test_string])
            
        # Instantiate Perceptron class
        perceptron = Perceptron(learning_rate=1, maxIter=20, l2_reg = l2_reg)

        # Training the Perceptron model using train data
        perceptron.fit(X_train, y_train)

        # Predicting the Perceptron model using test data
        y_pred = perceptron.predict(X_test)
        
        # Predicting the Perceptron model using train data
        x_pred = perceptron.predict(X_train)

        # Evaluate test data performance
        test_accuracy = np.mean(y_pred == y_test)
        
        # Evaluate train data performance
        train_accuracy = np.mean(x_pred == y_train)

        # Returns the calculated train and test accuracies
        return [train_accuracy, test_accuracy]
    
    # If there is a thirdClass, then we implement the One vs Rest Perceptron algorithm    
    else:
        
        # Define a dictionary to map string labels to their corresponding ids
        target_var_dict = {firstClass: 0, secondClass: 1, thirdClass: 2}

        y_train = np.array([target_var_dict[yi] for yi in y_train_string])

        # Define a dictionary to map string labels to their corresponding ids
        target_var_dict = {firstClass: 0, secondClass: 1, thirdClass: 2}

        y_test = np.array([target_var_dict[yi] for yi in y_test_string])

        
    # Instantiate OvRPerceptron class
    ovr_perceptron = OvRPerceptron(learning_rate=1, maxIter=20, l2_reg = l2_reg)

   # Training the OvRPerceptron model using train data
    ovr_perceptron.fit(X_train, y_train)

    # Predicting the OvRPerceptron model using test data
    y_pred = ovr_perceptron.predict(X_test)
    
    # Predicting the OvRPerceptron model using train data
    x_pred = ovr_perceptron.predict(X_train)

    # Evaluate test data performance
    test_accuracy = np.mean(y_pred == y_test)
    
    # Evaluate train data performance
    train_accuracy = np.mean(x_pred == y_train)
    
    # Return the accuracies.
    return [train_accuracy, test_accuracy]


def trainDataManipulation(train_data, className):
    
    # Filtering the train_data to remove any rows where the value in column 4 is equal to the className.
    train_data = train_data[train_data[4] != className]
    
    # Converting the filtered train_data into a numpy array.
    train_array = np.array(train_data)
    
    # Shuffling the rows of train_array randomly.
    random.shuffle(train_array)
    
    # Return the shuffled train_array.
    return train_array


def testDataManipulation(test_data, className):
    
    # Filtering the test_data to remove any rows where the value in column 4 is equal to the className.
    test_data = test_data[test_data[4] != className]
    
    # Converting the filtered test_data into a numpy array.
    test_array = np.array(test_data)
    
    # Shuffling the rows of test_array randomly.
    random.shuffle(test_array)
    
    # Return the shuffled test_array.
    return test_array


def activation(z):
    # For each element of `z`, if the element is greater than 0, the corresponding element in the predicted array will be 1.
    # Otherwise, it will be -1.
    return np.where(z>0,1,-1)

class Perceptron:
    
    def __init__(self, learning_rate, maxIter, l2_reg):
        
        # Initializing class attributes
        self.weights = None
        self.bias = None
        self.activation = activation # Setting the activation function
        self.learning_rate = learning_rate # Setting the learning rate
        self.maxIter = maxIter # Setting the number of maxIter for training
        self.l2_reg = l2_reg # Setting the L2 regularization parameter

    def fit(self, X, y):
        
        # Getting the number of features in the input data
        n_features = X.shape[1]
        
        # Initializing weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Iterating until the number of maxIter
        for iter in range(self.maxIter):
            
            # Traversing through the entire training set
            for i in range(len(X)):
                
                # Finding the dot product and adding the bias
                z = np.dot(X[i], self.weights) + self.bias
                # Passing through the activation function
                y_pred = self.activation(z)
                
                # Checking if the prediction was incorrect
                if y[i] * y_pred <= 0:
                    
                    #Updating weights and bias using the perceptron learning rule with L2 regularization
                    self.weights = (1 - 2 * self.l2_reg) * self.weights +  y[i] * X[i] * self.learning_rate
                    self.bias = self.bias + y[i] * self.learning_rate
                    
                else:
                    
                    # If the prediction was correct, applying only L2 regularization to weights and not updating the bias
                    self.weights= (1 - 2 * self.l2_reg) * self.weights
                    self.bias=self.bias
                
        return self.weights, self.bias
    
    def predict(self, X):
        
        # Finding the dot product and adding the bias  
        z = np.dot(X, self.weights) + self.bias 
        # Returning the predicted class based on the activation function applied to the dot product
        return self.activation(z)


class OvRPerceptron:
    def __init__(self, learning_rate, maxIter, l2_reg):
        
        # Initializing the OvRPerceptron object
        self.learning_rate = learning_rate
        self.maxIter = maxIter
        self.classifiers = [] # Initializing an empty list to hold the binary classifiers
        self.l2_reg = l2_reg

    def fit(self, X, y):
        
        # Training one binary classifier for each one vs rest classes
        for class_label in np.unique(y):
            # Converting y to binary labels
            y_binary = np.where(y == class_label, 1, -1)
            # Initializing a Perceptron object for each binary classification task
            perceptron = Perceptron(learning_rate=self.learning_rate, maxIter=self.maxIter, l2_reg=self.l2_reg)
            # Training the Perceptron object on the binary classification task
            perceptron.fit(X, y_binary)
            # Storing the trained classifier in a list
            self.classifiers.append(perceptron)

    def predict(self, X):

        y_pred = []
        for i in range(X.shape[0]):
            z_list = []
            # Computing the weighted sum of the inputs for each binary classifier
            for perceptron in self.classifiers:
                z = np.dot(X[i], perceptron.weights) + perceptron.bias
                z_list.append(z)
            # Predicting the class with the highest output value
            predicted_class = np.argmax(z_list)
            y_pred.append(predicted_class)
        # Converting the predicted class labels to a numpy array
        return np.array(y_pred)


    
# Loading the train and test data
train_data = pd.read_csv("train.data", delimiter=",", header=None)
test_data = pd.read_csv("test.data", delimiter=",", header=None)

print("\033[1m" + "***** Binary Classification Perceptron (Question 3) *****" + "\033[0m")
print("")
################## CLASS-1 VS CLASS-2 ##################

# Performing data manipulation on the training data
train_array = trainDataManipulation(train_data, "class-3")

# Performing data manipulation on the testing data
test_array = testDataManipulation(test_data, "class-3")

# Implementing a perceptron algorithm on the manipulated training and test data
accuracies = perceptronImpl(train_array, test_array, "class-1", "class-2", None, 0)

# Printing the accuracy of the classification between class-1 and class-2
print("Class-1 vs Class-2 Train Accuracy =", round(accuracies[0]*100, 2), "%")
print("Class-1 vs Class-2 Test Accuracy =", round(accuracies[1]*100, 2), "%")
print("")


################## CLASS-2 VS CLASS-3 ##################

# Performing data manipulation on the training data
train_array = trainDataManipulation(train_data, "class-1")

# Performing data manipulation on the testing data
test_array = testDataManipulation(test_data, "class-1")

# Implementing a perceptron algorithm on the manipulated training and test data
accuracies = perceptronImpl(train_array, test_array, "class-2", "class-3", None, 0)

# Printing the accuracy of the classification between class-2 and class-3
print("Class-2 vs Class-3 Train Accuracy =", round(accuracies[0]*100, 2), "%")
print("Class-2 vs Class-3 Test Accuracy =", round(accuracies[1]*100, 2), "%")
print("")


################## CLASS-1 VS CLASS-3 ##################

# Performing data manipulation on the training data
train_array = trainDataManipulation(train_data, "class-2")

# Performing data manipulation on the testing data
test_array = testDataManipulation(test_data, "class-2")

# Implementing a perceptron algorithm on the manipulated training and test data
accuracies = perceptronImpl(train_array, test_array, "class-1", "class-3", None, 0)

# Printing the accuracy of the classification between class-2 and class-3
print("Class-1 vs Class-3 Train Accuracy =", round(accuracies[0]*100, 2), "%")
print("Class-1 vs Class-3 Test Accuracy =", round(accuracies[1]*100, 2), "%")
print("")

print("\033[1m" + "***** Multi-Class Classification Perceptron + l2 Regularisation (Question 4 & 5) *****" + "\033[0m")
print("\033[1m" + "Note:" + "\033[0m", "l2 coefficient = 0 is the output for (Question 4)")
################## ONE VS REST APPROACH ##################
    
# Defining a list of L2 regularization values
l2_values = [0, 0.01, 0.1, 1, 10, 100]

# Iterating over each L2 value
for i in range(len(l2_values)):

    print("")
    print("\033[1m" + "l2 coefficient: " + "\033[0m", l2_values[i])

    # Converting train_data and test_data to numpy arrays and shuffling them
    train_array = np.array(train_data)
    test_array = np.array(test_data)
    random.shuffle(train_array)
    random.shuffle(test_array)

    # Calling the perceptronImpl function passing in the train and test arrays, class labels, and the current L2 value
    accuracies = perceptronImpl(train_array, test_array, "class-1", "class-2", "class-3", l2_values[i])
    
    # Printing the accuracy of the multi-class classifier
    print("Multi-Class Classifier Train Accuracy =", round(accuracies[0]*100, 2), "%")
    print("Multi-Class Classifier Test Accuracy =", round(accuracies[1]*100, 2), "%")
    print("")

