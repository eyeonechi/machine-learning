# adapted from Machine Learning Mastery by Jason Brownlee
# https://machinelearningmastery.com/

from random import randrange, seed

def zero_rule_regressor(train, test):
    output_values = [row[-1] for row in train]
    prediction = sum(output_values) / float(len(output_values))
    predicted = [prediction for i in range(len(test))]
    return predicted

def main():
    seed(1)
    train = [[10], [15], [12], [15], [18], [20]]
    test = [[None], [None], [None], [None]]
    predictions = zero_rule_regressor(train, test)
    print(predictions)

if __name__ == '__main__':
    main()
