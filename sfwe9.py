import numpy as np
import struct
import os

# Function to read IDX file format
def read_idx(filename):
    with open(filename, 'rb') as f:
        magic_number = f.read(4)
        magic_number = struct.unpack('>I', magic_number)[0]

        if magic_number == 2051:
            num_images = struct.unpack('>I', f.read(4))[0]
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
            return data.astype(np.float32) / 255.0
        elif magic_number == 2049:  # Labels
            num_labels = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data

# Load the MNIST dataset from the specified path
def load_data(path):
    X_train_path = os.path.join(path, 'train-images.idx3-ubyte')
    y_train_path = os.path.join(path, 'train-labels.idx1-ubyte')
    X_test_path = os.path.join(path, 't10k-images.idx3-ubyte')
    y_test_path = os.path.join(path, 't10k-labels.idx1-ubyte')

    X_train = read_idx(X_train_path).reshape(-1, 28, 28, 1)
    y_train = read_idx(y_train_path)
    X_test = read_idx(X_test_path).reshape(-1, 28, 28, 1)
    y_test = read_idx(y_test_path)

    return X_train, y_train, X_test, y_test

# Define the Convolutional Layer
def convolve(input_image, filter_weights, bias):
    input_height, input_width = input_image.shape
    filter_height, filter_width = filter_weights.shape

    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1
    output_map = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = input_image[i:i + filter_height, j:j + filter_width]
            output_map[i, j] = np.sum(region * filter_weights) + bias

    return output_map

# Define the Max Pooling Layer
def max_pooling(input_map, pool_size):
    input_height, input_width = input_map.shape
    output_height = input_height // pool_size
    output_width = input_width // pool_size
    output_map = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = input_map[i * pool_size:(i + 1) * pool_size, j * pool_size:(j + 1) * pool_size]
            output_map[i, j] = np.max(region)

    return output_map

# Implementing a CNN
def cnn_forward(input_images, filter_weights, biases, pool_size):
    pooled_maps = []

    for image in input_images:
        convolved_map = convolve(image.reshape(28, 28), filter_weights, biases)
        pooled_map = max_pooling(convolved_map, pool_size)
        pooled_maps.append(pooled_map)

    return np.array(pooled_maps)

# Flatten the pooled maps for the fully connected layer
def flatten(input_maps):
    return input_maps.reshape(input_maps.shape[0], -1)

# Initialize weights and biases
filter_weights = np.random.randn(3, 3)
biases = np.random.randn()
pool_size = 2

# Load dataset
path = 'C:/Users/kavit/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1'
X_train, y_train, X_test, y_test = load_data(path)

# Perform the CNN forward pass
pooled_train = cnn_forward(X_train, filter_weights, biases, pool_size)
pooled_test = cnn_forward(X_test, filter_weights, biases, pool_size)

# Flatten the pooled maps for a fully connected layer
flattened_train = flatten(pooled_train)
flattened_test = flatten(pooled_test)

# fully connected layer
def fully_connected(input_data, weights, bias):
    return np.dot(input_data, weights) + bias


# Initialize weights and bias for  fully connected layer
num_classes = 10  # Digits 0-9
fc_weights = np.random.randn(flattened_train.shape[1], num_classes)
fc_bias = np.random.randn(num_classes)


# Train the model
def train(X, y, X_test, y_test, num_epochs=5, learning_rate=0.01):
    global fc_weights, fc_bias  
    for epoch in range(num_epochs):
        logits = fully_connected(X, fc_weights, fc_bias)

        # Evaluate accuracy on the training set
        train_accuracy = evaluate(X, y)
        test_accuracy = evaluate(X_test, y_test)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%")


# Evaluate the model
def evaluate(X, y):
    logits = fully_connected(X, fc_weights, fc_bias)
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy


# Train the model
train(flattened_train, y_train, flattened_test, y_test)

# Final evaluation after training
final_accuracy = evaluate(flattened_test, y_test)
print(f"Final Test Accuracy: {final_accuracy * 100:.2f}%")
