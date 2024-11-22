import numpy as np
import pandas as pd

def load_data(filename):
    return pd.read_csv(filename)

def check_pass_fail(grades):
    return int(all(grade >= 50 for grade in grades))

def prepare_data(df):
    X = df[[f'Course{i}' for i in range(1, 7)]].values
    y = np.array([check_pass_fail(row) for row in X])
    return X, y

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.hidden_layer = self.relu(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output_layer = self.softmax(np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output)
        return self.output_layer

    def backward(self, X, y):
        num_samples = X.shape[0]
        output_error = self.output_layer
        output_error /= num_samples

        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_error[self.hidden_layer <= 0] = 0

        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_layer.T, output_error)
        self.bias_output -= self.learning_rate * np.sum(output_error, axis=0)
        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_error)
        self.bias_hidden -= self.learning_rate * np.sum(hidden_error, axis=0)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if (epoch + 1) % 50 == 0:
                accuracy = self.calculate_accuracy(X, y)
                print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}%")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def calculate_accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100

filename = 'students_grades.csv'
df = load_data(filename)
X, y = prepare_data(df)

input_size = X.shape[1]
hidden_size = 5
output_size = 2
learning_rate = 0.01
epochs = 500

mlp = MLP(input_size, hidden_size, output_size, learning_rate)
mlp.train(X, y, epochs)

# Final accuracy output
accuracy = mlp.calculate_accuracy(X, y)
print(f"Final Model accuracy: {accuracy:.2f}%")
