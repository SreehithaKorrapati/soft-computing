import numpy as np

class MLPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.05, epochs=2000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights1 = np.random.rand(input_size, hidden_size) * 2 - 1
        self.weights2 = np.random.rand(hidden_size, output_size) * 2 - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets):
        for epoch in range(self.epochs):
            inputs, targets = np.array(inputs), np.array(targets)
            hidden_layer_outputs, outputs = self.feedforward(inputs)
            errors = targets - outputs
            delta2 = errors * self.sigmoid_derivative(outputs)
            delta1 = delta2.dot(self.weights2.T) * self.sigmoid_derivative(hidden_layer_outputs)
            self.weights2 += self.learning_rate * hidden_layer_outputs.T.dot(delta2)
            self.weights1 += self.learning_rate * inputs.T.dot(delta1)
# FOR EVERY 500
            if epoch % 500 == 0:
                loss = np.mean(np.square(errors))
                print(f"Epoch {epoch}, Loss: {loss}")
                for input, target in zip(inputs, targets):
                    output = self.feedforward(input)[1]
                    print(f"Input: {input}, Target: {target}, Output: {output}")
                print("------")

    def feedforward(self, inputs):
        hidden_inputs = np.dot(inputs, self.weights1)
        hidden_layer_outputs = self.sigmoid(hidden_inputs)
        final_inputs = np.dot(hidden_layer_outputs, self.weights2)
        final_outputs = self.sigmoid(final_inputs)
        return hidden_layer_outputs, final_outputs

epochs = 2000
mlp = MLPerceptron(input_size=2, hidden_size=2, output_size=1, learning_rate=0.05, epochs=epochs)

inputs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
targets = np.array([[1], [-1], [-1], [1]])

mlp.train(inputs, targets)
# Testing
print("\nTesting after training:")
for input, target in zip(inputs, targets):
    output = mlp.feedforward(input)[1] 
    print(f"Input: {input}, Target: {target}, Output: {output}")
