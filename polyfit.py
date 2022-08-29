import numpy as np
import matplotlib.pyplot as plt

# Return fitted model parameters to the dataset at datapath for each choice in degrees.
# Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
# Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
# coefficients when fitting a polynomial of d = degrees[i].
def main(datapath, degrees):
    paramFits = []

    # fill in
    # read the input file, assuming it has two columns, where each row is of the form [x y] as
    # in poly.txt.
    # iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    # for the model parameters in each case. Append the result to paramFits each time.

    with open(datapath, 'r') as file:
        lines = file.readlines()
        xlist = []
        ylist = []
        for line in lines:
            digits = line.split(' ')
            x = float(digits[0])
            y = float(digits[1])
            xlist.append(x)
            ylist.append(y)

    for degree in degrees:
        X = feature_matrix(xlist, degree)
        B = least_squares(X, ylist)
        paramFits.append(B)

    #print(X)
    #print(xlist)
    plt.scatter(xlist, ylist, color='black', label='Data')
    xlist.sort()
    y1 = list(map(lambda x: x * paramFits[0][0] + paramFits[0][1], xlist))
    y2 = list(map(lambda x: x ** 2 * paramFits[1][0] + x * paramFits[1][1] + paramFits[1][2], xlist))
    y3 = list(map(lambda x: x ** 3 * paramFits[2][0] + x ** 2 * paramFits[2][1] + x * paramFits[2][2] + paramFits[2][3], xlist))
    y4 = list(map(lambda x: x ** 4 * paramFits[3][0] + x ** 3 * paramFits[3][1] + x ** 2 * paramFits[3][2] + x * paramFits[3][3] + paramFits[3][4], xlist))
    y5 = list(map(lambda x: x ** 5 * paramFits[4][0] + x ** 4 * paramFits[4][1] + x ** 3 * paramFits[4][2] + x ** 2 * paramFits[4][3] + x * paramFits[4][4] + paramFits[4][5], xlist))


    plt.plot(xlist, y1, color='blue', label='D1')
    plt.plot(xlist, y2, color='yellow', label='D2')
    plt.plot(xlist, y3, color='green', label='D3')
    plt.plot(xlist, y4, color='red', label='D4')
    plt.plot(xlist, y5, color='purple', label='D5')
    plt.legend()
    plt.show()
    return paramFits


# Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
# samples in x.
# Input: x as a list of the independent variable samples, and d as an integer.
# Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
# for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):

    # fill in
    # There are several ways to write this function. The most efficient would be a nested list comprehension
    # which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    X = []
    i = 0
    for item in x:
        X.append([])
        for degree in range(d + 1):
            X[i].insert(0, item ** degree)
        i = i + 1

    return X


# Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
# Input: X as a list of features for each sample, and y as a list of target variable samples.
# Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)

    # fill in
    # Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    B = np.linalg.inv(X.T @ X) @ X.T @ y

    #print(B)

    return B


if __name__ == "__main__":
    datapath = "poly.txt"
    degrees = [1, 2, 3, 4, 5]

    paramFits = main(datapath, degrees)
    print(paramFits)
