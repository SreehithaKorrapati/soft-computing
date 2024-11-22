import numpy as np

def initialize_weights(input_dim, num_units):
    return np.random.normal(loc=0.5, scale=0.1, size=(num_units, input_dim))

def train_ksom(dataset, num_units, max_iterations, initial_learning_rate, initial_neighborhood, reduction_interval):
    input_dim = dataset.shape[1]
    weights = initialize_weights(input_dim, num_units)

    for iteration in range(max_iterations):
        learning_rate = initial_learning_rate
        if iteration > reduction_interval:
            learning_rate *= (0.5 ** ((iteration - reduction_interval) // reduction_interval))
        neighborhood = max(1, int(initial_neighborhood * (0.5 ** (iteration // reduction_interval))))

        for input_vector in dataset:
            distances = np.sum((input_vector - weights) ** 2, axis=1)
            winner_index = np.argmin(distances)
            for j in range(weights.shape[0]):
                if abs(j - winner_index) <= neighborhood:
                    weights[j] += learning_rate * (input_vector - weights[j])

    return weights

dataset = np.genfromtxt('student_grades.csv', delimiter=',', skip_header=1, usecols=range(1, 7))
dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0)

print("Dataset shape:", dataset.shape)

num_units = 5
max_iterations = 500
initial_learning_rate = 0.5
initial_neighborhood = 5
reduction_interval = 100

weights = train_ksom(dataset, num_units, max_iterations, initial_learning_rate, initial_neighborhood, reduction_interval)

clusters = np.zeros(dataset.shape[0])
for i, input_vector in enumerate(dataset):
    distances = np.sum((input_vector - weights) ** 2, axis=1)
    clusters[i] = np.argmin(distances)

# OUTPUT
unique, counts = np.unique(clusters, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"Cluster {int(cluster)}: {count} students")
