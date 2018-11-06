# adapted from Machine Learning Mastery by Jason Brownlee
# https://machinelearningmastery.com/

from csv import reader
from math import sqrt
from random import randrange, seed

# Load a CSV file
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        header = csv_reader.next()
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column values to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Split a dataset into train and test set
def train_test_split(dataset, split):
    train = []
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

# Calculates the mean value of a list of numbers
def mean(values):
    return sum(values) / float(len(values))

# Calculates the variance of a list of numbers
def variance(values, mean):
    return sum([(x - mean) ** 2 for x in values])

# Calculates the covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

# Calculates coefficients
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    mean_x, mean_y = mean(x), mean(y)
    var_x, var_y = variance(x, mean_x), variance(y, mean_y)
    covar = covariance(x, mean_x, y, mean_y)
    slope = covar / var_x
    intercept = mean_y - slope * mean_x

    print('x stats: mean=%.3f variance=%.3f' % (mean_x, var_x))
    print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y))
    print('covariance=%.3f' % (covar))
    print('coefficients: slope=%.3f, intercept=%.3f' % (slope, intercept))

    return [slope, intercept]

# Simple Linear Regression
def simple_linear_regression(train, test):
    predictions = []
    slope, intercept = coefficients(train)
    for row in test:
        yhat = slope * row[0] + intercept
        predictions.append(yhat)
    return predictions

# Calculates root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(predicted)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# Evaluate regression algorithm on training dataset
def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split)
    test_set = []
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    print('predicted:')
    print(predicted)
    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual, predicted)
    return rmse

# 0: Load and prepare data
# 1: Calculate mean and variance
# 2: Calculate covariance
# 3: Estimate coefficients
# 4: Make predictions
# 5: Evaluate algorithm
def main():
    # dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
    seed(1)
    filename = '../data/insurance.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    split = 0.6
    rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
    print('RMSE: %.3f' % (rmse))

if __name__ == '__main__':
    main()
