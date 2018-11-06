# adapted from Machine Learning Mastery by Jason Brownlee
# https://machinelearningmastery.com/

from random import randrange, seed

# Generate random predictions
def random_prediction(train, test):
    output_values = [row[-1] for row in train]
    unique = list(set(output_values))
    predicted = []
    for row in test:
        index = randrange(len(unique))
        predicted.append(unique[index])
    return predicted

def main():
    seed(1)
    train = [[0], [1], [0], [1], [0], [1]]
    test = [[None], [None], [None], [None]]
    predictions = random_prediction(train, test)
    print(predictions)

if __name__ == '__main__':
    main()
