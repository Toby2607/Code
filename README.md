# NN model code and NN + PSO model code
## NN code
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Read data and 
data_dir = "/content/drive/MyDrive/Data"  
img_size = 64  # resize image

X = []  
y = []

# Read images
for label, folder in enumerate(["cat", "dog"]):
    folder_path = os.path.join(data_dir, folder)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label)

X = np.array(X)  # Convert to numpy array
y = np.array(y)

# Standardize data to [0, 1]
X = X / 255.0


# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Neural Network 
def create_nn_model(input_shape, dense_neurons=128, dropout_rate=0.2, learning_rate=0.001):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(dense_neurons, activation='relu'))
    model.add(layers.Dense(dense_neurons, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

learning_rate = 0.0001
model = create_nn_model(X_train.shape[1:], learning_rate=learning_rate)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluation
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

model.save('nn_model.h5')
print("Model is saved")

# Evaluation plot
plt.figure(figsize=(12, 6))
# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read data and preprocessing
data_dir = "/content/drive/MyDrive/Data"  
img_size = 64 #resize image
X = []
y = []

for label, folder in enumerate(["cat", "dog"]):
    folder_path = os.path.join(data_dir, folder)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label)

X = np.array(X)
y = np.array(y)
X = X / 255.0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build NN model
def create_nn_model(input_shape, dense_neurons, dropout_rate, learning_rate, num_layers, activation):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    for _ in range(num_layers):
        model.add(layers.Dense(dense_neurons, activation=activation))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# PSO configuration
DIMENSIONS = 5
SWARM_SIZE = 10
INFORMANTS = 3
NUM_GENERATIONS = 20
W = 0.729
C1 = 1.49
C2 = 1.49
MIN_BOUNDARY = [8, 0.1, 1e-5, 1, 0]
MAX_BOUNDARY = [256, 0.5, 1e-2, 3, 2]
activation_functions = ['relu', 'tanh', 'elu']
desired_precision = 1e-5


# Fitness function for PSO algorithm
def fitness_function(position):
    dense_neurons = int(position[0])
    dropout_rate = float(position[1])
    learning_rate = float(position[2])
    num_layers = int(position[3])
    activation_idx = int(position[4])
    dense_neurons = max(8, min(dense_neurons, 256))
    dropout_rate = max(0.1, min(dropout_rate, 0.5))
    learning_rate = max(1e-5, min(learning_rate, 1e-2))
    num_layers = max(1, min(num_layers, 3))
    activation = activation_functions[max(0, min(activation_idx, len(activation_functions)-1))]
    model = create_nn_model(X_train.shape[1:],dense_neurons=dense_neurons,dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        num_layers=num_layers,activation=activation)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_val, y_val))
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    return 1 - val_acc

class Particle:
    def __init__(self):
        self.position = [random.uniform(MIN_BOUNDARY[i], MAX_BOUNDARY[i]) for i in range(DIMENSIONS)]
        self.velocity = [random.uniform(-1, 1) for _ in range(DIMENSIONS)]
        self.fitness = fitness_function(self.position)
        self.best_position = list(self.position)
        self.best_fitness = self.fitness
        self.informants = random.sample(range(SWARM_SIZE), INFORMANTS)
        self.group_best_position = list(self.position)
        self.group_best_fitness = self.fitness

    def update_velocity(self):
        for d in range(DIMENSIONS):
            r1, r2 = random.random(), random.random()
            cognitive = C1 * r1 * (self.best_position[d] - self.position[d])
            social = C2 * r2 * (self.group_best_position[d] - self.position[d])
            self.velocity[d] = W * self.velocity[d] + cognitive + social

    def update_position(self):
        for d in range(DIMENSIONS):
            self.position[d] += self.velocity[d]
            self.position[d] = max(MIN_BOUNDARY[d], min(MAX_BOUNDARY[d], self.position[d]))
        self.fitness = fitness_function(self.position)

    def update_group_best(self, swarm):
        best_informant = min(self.informants, key=lambda i: swarm[i].best_fitness)
        if swarm[best_informant].best_fitness < self.group_best_fitness:
            self.group_best_fitness = swarm[best_informant].best_fitness
            self.group_best_position = list(swarm[best_informant].best_position)

import matplotlib.pyplot as plt

# Implement model
swarm = [Particle() for _ in range(SWARM_SIZE)]
global_best = min(swarm, key=lambda p: p.best_fitness)
global_best_position = list(global_best.best_position)
global_best_fitness = global_best.best_fitness

fitness_history = []

for gen in range(NUM_GENERATIONS):
    for particle in swarm:
        particle.update_group_best(swarm)
        particle.update_velocity()
        particle.update_position()
        if particle.fitness < particle.best_fitness:
            particle.best_fitness = particle.fitness
            particle.best_position = list(particle.position)

    best_particle = min(swarm, key=lambda p: p.best_fitness)
    if best_particle.best_fitness < global_best_fitness:
        global_best_fitness = best_particle.best_fitness
        global_best_position = list(best_particle.best_position)

    fitness_history.append(1 - global_best_fitness)
    print(f"Generation {gen+1}: Validation ccuracy = {1 - global_best_fitness:.4f}")

    if global_best_fitness < desired_precision:
        print("Desired precision reached.")
        break
# The most optimal hyperparameter setting 
print("\nOptimization Complete!")
print(f"Neurons: {int(global_best_position[0])}")
print(f"Dropout: {global_best_position[1]:.2f}")
print(f"Learning rate: {global_best_position[2]:.6f}")
print(f"Number of layers: {int(global_best_position[3])}")
print(f"Activation function: {activation_functions[int(global_best_position[4])]}")
print(f"Best validation accuracy: {1 - global_best_fitness:.4f}")

# FITNESS graph
plt.plot(range(1, NUM_GENERATIONS + 1), fitness_history, marker='o')
plt.title('Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()
