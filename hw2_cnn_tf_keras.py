#!/usr/bin/env python 3

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Set the device to GPU if available, otherwise use CPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    device = "GPU"
else:
    device = "CPU"

class NeuralNetwork(keras.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.pool = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(56, (3, 3), activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(56, activation='relu')
        self.fc2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

def run(model, loader, optimizer, mode='train'):
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    if mode == 'train':
        model.train()
    else:
        model.eval()

    tot_loss = 0.0
    tot_correct = 0
    tot_samples = 0

    for batch, (X, y) in enumerate(loader):
        X, y = X.numpy(), y.numpy()

        with tf.GradientTape() as tape:
            logits = model(X, training=mode == 'train')
            loss_value = loss_fn(y, logits)

        if mode == 'train':
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        tot_loss += loss_value.numpy()
        predictions = np.argmax(logits, axis=1)
        correct = np.sum(predictions == y)
        tot_correct += correct
        tot_samples += X.shape[0]

        if mode == 'train' and batch % 100 == 99:
            print(f"[{tot_samples:>5d}/{len(loader.dataset):>5d}] loss: {loss_value.numpy():>5f}, accuracy: {(correct / X.shape[0] * 100):>2.4f}%")

    accuracy = tot_correct / tot_samples * 100
    loss = tot_loss / len(loader.dataset)
    print(f'{mode.upper():>6s}: Average loss: {loss:>5f}, Accuracy: {tot_correct:>5d}/{tot_samples:>5d} ({accuracy:>2.4f}%)')
    return accuracy

def main():
    lr = 0.001
    epochs = 10
    batch_size = 32
    np.random.seed(7)
    tf.random.set_seed(7)

    # Load the Fashion MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    # Preprocess the data
    train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

    train_set = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(48000).batch(batch_size)
    valid_set = tf.data.Dataset.from_tensor_slices((train_images[48000:], train_labels[48000:])).batch(batch_size)
    test_set = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    # Class labels
    labels = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Initialize model, optimizer
    model = NeuralNetwork()
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    total_params = model.count_params()
    print(f"Number of trainable parameters: {total_params}")

    train_accuracies = []
    valid_accuracies = []
    for epoch in range(1, epochs + 1):
        print(f'epoch {epoch} / {epochs}:')
        train_acc = run(model, train_set, optimizer, 'train')
        valid_acc = run(model, valid_set, None, 'valid')
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

    x = range(1, epochs + 1)
    plt.figure(1)
    plt.plot(x, train_accuracies, 's-')
    plt.plot(x, valid_accuracies, 'o--')
    plt.legend(['train', 'validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.grid()
    plt.savefig('accuracies.png')

    # Evaluate the accuracy on the test set
    test_accuracy = run(model, test_set, None, 'test')
    print(f'Accuracy on Test set is {test_accuracy:>2.4f}%')

    # Examples of misclassification
    sampler = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    misclass = [None] * 10
    found = [False] * 10
    cnt = 0
    for X, y in sampler:
        X = X.numpy()
        output = model(X[tf.newaxis, ...])
        pred = tf.argmax(output, axis=1).numpy()[0]
        y = y.numpy()
        if pred != y and not found[y]:
            found[y] = True
            cnt += 1
            misclass[y] = (pred, X)
            if cnt == 10:
                break

    plt.figure(figsize=(12, 6))  

    for y, (pred, X) in enumerate(misclass):
        plt.subplot(2, 5, y + 1)
        plt.imshow(X.reshape((28, 28)))
        plt.title(f'Truth:{labels[y]:>5s}\nPredicted:{labels[pred]:>5s}')

    plt.tight_layout()  
    plt.savefig('examples.png')

if __name__ == "__main__":
    main()
