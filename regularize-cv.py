import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Part 1
# Function that normalizes features in training set to zero mean and unit variance.
# Input: training data X_train
# Output: the normalized version of the feature matrix: X, the mean of each column in
# training set: trn_mean, the std dev of each column in training set: trn_std.
def normalize_train(X_train):
    X_mat = np.array(X_train)
    X_avgs = list(np.mean(X_mat, axis=0))
    X_stds = list(np.std(X_mat, axis=0))

    for sublist in X_train:
        for i in range(9):
            sublist[i] = (sublist[i] - X_avgs[i]) / X_stds[i]


    return X_train, X_avgs, X_stds


# Part 2
# Function that normalizes testing set according to mean and std of training set
# Input: testing data: X_test, mean of each column in training set: trn_mean, standard deviation of each
# column in training set: trn_std
# Output: X, the normalized version of the feature matrix, X_test.
def normalize_test(X_test, trn_mean, trn_std):

    # fill in
    for sublist in X_test:
        for i in range(9):
            sublist[i] = (sublist[i] - trn_mean[i]) / trn_std[i]

    return X_test


# Part 3
# Function to return a numpy array generated with `np.logspace` with a length
# of 51 starting from 1E^-1 and ending at 1E^3
def get_lambda_range():

    # fill in
    lmbda = np.logspace(start=-1, stop=3, num=51)


    return lmbda


# Part 4
# Function that trains a ridge regression model on the input dataset with lambda=l.
# Input: Feature matrix X, target variable vector y, regularization parameter l.
# Output: model, a numpy object containing the trained model.
def train_model(X, y, l):

    # fill in
    model = Ridge(alpha=l, fit_intercept=True).fit(X, y)

    return model

#Part 5
#Function that trains a Lasso regression model on the input dataset with lambda=l.
#Input: Feature matrix X, target variable vector y, regularization parameter l.
#Output: model, a numpy object containing the trained model
def train_model_lasso(X,y,l):

    #fill in
    model = Lasso(alpha=l, fit_intercept=True).fit(X,y)

    return model

# Part 6
# Function that calculates the mean squared error of the model on the input dataset.
# Input: Feature matrix X, target variable vector y, numpy model object
# Output: mse, the mean squared error
def error(X, y, model):

    # Fill in
    #Add a column of 1s to our X matrix

    X_matrix = np.array(X)
    rows = X_matrix.shape[0]
    ones = np.ones((rows, 1))
    X_matrix = np.concatenate([X_matrix, ones], axis=1)

    # Had a weird error going on where Ridge returned a model that is a list IN a list, Lasso returns just a list
    # Fixed with the following if statement
    if (model.coef_.shape) == (1, 9):
        modelCoeffs = model.coef_[0]
    else:
        modelCoeffs = model.coef_
    model_matrix = np.concatenate([modelCoeffs, model.intercept_], axis=0)

    model_matrix = model_matrix.T

    Y_Predict = X_matrix @ model_matrix


    mse = mean_squared_error(y_true=y, y_pred=Y_Predict)

    return mse


def main():
    #Importing dataset
    diamonds = pd.read_csv('diamonds.csv')

    #Feature and target matrices
    X = diamonds[['carat', 'depth', 'table', 'x', 'y', 'z', 'clarity', 'cut', 'color']]
    y = diamonds[['price']]

    #Training and testing split, with 25% of the data reserved as the test set
    X = X.to_numpy()
    y = y.to_numpy()
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)

    # Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)

    X_test = normalize_test(X_test, trn_mean, trn_std)

    # Define the range of lambda to test
    lmbda = get_lambda_range()
    #lmbda = [1,100]
    MODEL = []
    MSE = []
    for l in lmbda:
        # Train the regression model using a regularization parameter of l
        model = train_model(X_train, y_train, l)
        #print(model.intercept_)
        # Evaluate the MSE on the test set
        mse = error(X_test, y_test, model)
        #print(mse)

        # Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)
        print('*************')
        print(model)
        print(model.coef_)
        #print('****************')
    # Part 6
    # Plot the MSE as a function of lmbda
    plt.plot(lmbda, MSE)
    plt.title('MSE as function of lambda')
    plt.xlabel('Lambda Values')
    plt.ylabel('Mean Sq Error Values')
    plt.show()

    # Part 7
    # Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE))  # fill in
    [lmda_best, MSE_best, model_best] = [lmbda[ind], MSE[ind], MODEL[ind]]
    #evaluate the best model for
    #0.25 carat, 3 cut, 3 color, 5 clarity, 60 depth, 55 table, 4 x, 3 y, 2 z diamond (Use the Ridge regression model `train_model`)
    # NOTE: Make sure to normalize the given data
    myDiamond = [0.25, 3, 3, 5, 60, 55, 4, 3, 2]
    #Normalize:
    for i in range(9):
        myDiamond[i] = (myDiamond[i] - trn_mean[i]) / trn_std[i]

    myDiamond.append(1)
    myDiamond_M = np.array(myDiamond)
    if (model_best.coef_.shape) == (1, 9):
        modelCoeffs = model_best.coef_[0]
    else:
        modelCoeffs = model_best.coef_
    Coeffs = np.concatenate([modelCoeffs, model_best.intercept_])
    myPrice = myDiamond_M @ Coeffs.T
    #print('my price is ')
    #print(myPrice)

    print(
        "Best lambda tested is "
        + str(lmda_best)
        + ", which yields an MSE of "
        + str(MSE_best)
    )


    return model_best


if __name__ == "__main__":
    model_best = main()
    # We use the following functions to obtain the model parameters instead of model_best.get_params()
    print(model_best.coef_)
    print(model_best.intercept_)
