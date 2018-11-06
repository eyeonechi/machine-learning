# adapted from Machine Learning Mastery by Jason Brownlee
# https://machinelearningmastery.com/

from random import randrange, seed

def zero_rule_classifier(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted

def main():
    seed(1)
    train = [['0'], ['0'], ['0'], ['0'], ['1'], ['1']]
    test = [[None], [None], [None], [None]]
    predictions = zero_rule_classifier(train, test)
    print(predictions)

if __name__ == '__main__':
    main()
