import numpy as np

class BPNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights1 = 2 * np.random.random((input_size, hidden_size)) - 1
        self.weights2 = 2 * np.random.random((hidden_size, output_size)) - 1
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y):
        for epoch in range(self.epochs):
            hidden_layer_activation = np.dot(X, self.weights1) + self.bias1
            hidden_layer_output = self.sigmoid(hidden_layer_activation)
            output_layer_activation = np.dot(hidden_layer_output, self.weights2) + self.bias2
            predicted_output = self.sigmoid(output_layer_activation)

            error = y - predicted_output
            d_predicted_output = error * self.sigmoid_derivative(predicted_output)
            error_hidden_layer = d_predicted_output.dot(self.weights2.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)

            self.weights2 += hidden_layer_output.T.dot(d_predicted_output) * self.learning_rate
            self.bias2 += np.sum(d_predicted_output, axis=0, keepdims=True) * self.learning_rate
            self.weights1 += X.T.dot(d_hidden_layer) * self.learning_rate
            self.bias1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

            if epoch % 100 == 0:
                accuracy = self.calculate_accuracy(X, y)
                print(f'Epoch {epoch}: Accuracy: {accuracy * 100:.2f}%')

    def calculate_accuracy(self, X, y):
        predictions = self.predict(X)
        predicted_classes = (predictions > 0.5).astype(int)
        return np.mean(predicted_classes == y)

    def predict(self, X):
        hidden_layer_activation = np.dot(X, self.weights1) + self.bias1
        hidden_layer_output = self.sigmoid(hidden_layer_activation)
        output_layer_activation = np.dot(hidden_layer_output, self.weights2) + self.bias2
        predicted_output = self.sigmoid(output_layer_activation)
        return predicted_output

def load_dataset(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    X = data[:, 1:7]
    y = np.where(np.any(X < 50, axis=1), 0, 1).reshape(-1, 1)
    return X, y

X, y = load_dataset('students_grades.csv')
max_values = np.max(X, axis=0)
if np.any(max_values == 0):
    raise ValueError("Max values cannot be zero for normalization.")
X = X / max_values

input_size = X.shape[1]
hidden_size = 4
output_size = 1

model = BPNN(input_size, hidden_size, output_size, learning_rate=0.01, epochs=500)
model.train(X, y)
predictions = model.predict(X)
predicted_classes = (predictions > 0.5).astype(int)
accuracy = np.mean(predicted_classes == y)

for i, prediction in enumerate(predictions):
    result = "Pass" if prediction > 0.5 else "Fail"
    print(f"Student {i + 1}: {result} (Prediction: {prediction[0]})")

print(f"Final Accuracy: {accuracy * 100:.2f}%")
