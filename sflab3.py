import random

def read_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            parts = line.strip().split(',')
            scores = [float(x) for x in parts[1:]]
            data.append(scores)
            if any(score < 50 for score in scores):
                labels.append(0)
            else:
                labels.append(1)
    return data, labels


class Perceptron:
    def __init__(self, input_size, learning_rate=1.0):
        self.weights = [random.uniform(-1.5, 1.0) for _ in range(input_size + 1)]
        self.learning_rate = learning_rate

    def predict(self, x):
        activation = self.weights[0]  # bias
        for i in range(len(x)):
            activation += self.weights[i + 1] * x[i]
        return 1 if activation >= 0 else 0

    def train(self, X, y, epochs):
        for _ in range(epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights[0] += self.learning_rate * error
                for j in range(len(X[i])):
                    self.weights[j + 1] += self.learning_rate * error * X[i][j]


def evaluate_perceptron(X, y, model):
    correct = 0
    for i in range(len(X)):
        prediction = model.predict(X[i])
        if prediction == y[i]:
            correct += 1
    accuracy = correct / len(X)
    return accuracy


def main():
    data, labels = read_data('students_grades.csv')
    perceptron = Perceptron(input_size=len(data[0]))

    perceptron.train(data, labels, epochs=100)

    accuracy = evaluate_perceptron(data, labels, perceptron)
    print(f"Perceptron accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    data, labels = read_data('students_grades.csv')
    perceptron = Perceptron(input_size=len(data[0]))

    perceptron.train(data, labels, epochs=100)

    accuracy = evaluate_perceptron(data, labels, perceptron)
    print(f"Perceptron accuracy: {accuracy * 100:.2f}%")