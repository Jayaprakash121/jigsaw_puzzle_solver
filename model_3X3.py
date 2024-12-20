import os 
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

ground_truth = pd.read_csv("animal_3_3_jigsaw.csv")
ground_truth.head()

for row in range(3):
    for col in range(3):
        ground_truth[f'piece_{row}_{col}'] = ground_truth[f'piece_{row}_{col}'].map({'(0, 0)':0, '(0, 1)':1, '(0, 2)':2, '(1, 0)':3, '(1, 1)':4, '(1, 2)':5, '(2, 0)':6, '(2, 1)':7, '(2, 2)':8})

def split_images(image_name, image_dir, grid_size, ik):
    image_path = os.path.join(image_dir, image_name)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (grid_size*64, grid_size*64))
    height, width, _ = img.shape
    piece_height, piece_width = height // grid_size, width // grid_size
    
    correct_positions = ground_truth[ground_truth['image_name'] == image_name].iloc[0]
    pieces = []
    piece_positions = []
    for row in range(grid_size):
        for col in range(grid_size):
            piece = img[row * piece_height:(row + 1) * piece_height,
                        col * piece_width:(col + 1) * piece_width]
            ik += 1
            pieces.append(piece)
            piece_positions.append(correct_positions[f'piece_{row}_{col}'])
    
    return pieces, np.array(piece_positions), ik
            
image_dir = 'animal_3_3_jigsaw/'

print("training pieces started")
a = 0
training_pieces = []
training_positions = []
for i in range(6000):
    image_name = f'{i}.jpg'
    pc, pcp, v = split_images(image_name, image_dir, 3, a)
    a = v
    training_pieces.append(pc)
    training_positions.append(pcp)


training_pieces = np.array(training_pieces)
training_positions = np.array(training_positions)

training_pieces = training_pieces.reshape(-1, 64, 64, 3)
training_positions = training_positions.reshape(-1)

# For simplicity, assume the original labels are indices, and we create one-hot encodings
num_samples = 6000*9
num_classes = 9

# Convert labels to one-hot encoding
training_positions_one_hot = np.zeros((num_samples, num_classes))
for i, label in enumerate(training_positions):
    training_positions_one_hot[i, label] = 1

print("validating pieces started")
a = 0
validating_pieces = []
validating_positions = []
for i in range(6000, 8000):
    image_name = f'{i}.jpg'
    pc, pcp, v = split_images(image_name, image_dir, 3, a)
    a = v
    validating_pieces.append(pc)
    validating_positions.append(pcp)

validating_pieces = np.array(validating_pieces)
validating_positions = np.array(validating_positions)

validating_pieces = validating_pieces.reshape(-1, 64, 64, 3)
validating_positions = validating_positions.reshape(-1)


# For simplicity, assume the original labels are indices, and we create one-hot encodings
num_samples = 2000*9
num_classes = 9

# Convert labels to one-hot encoding
validating_positions_one_hot = np.zeros((num_samples, num_classes))
for i, label in enumerate(validating_positions):
    validating_positions_one_hot[i, label] = 1

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(9, activation='softmax') 
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

print("training the model")
model = build_model()
history = model.fit(
    training_pieces,  
    training_positions_one_hot,  
    validation_data=(validating_pieces, validating_positions_one_hot),  
    epochs=6,
    batch_size=250,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
)

'''
predictions = model.predict(training_pieces)

threshold = 0.5
predictions_binary = (predictions >= threshold).astype(int)

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(training_positions_one_hot, predictions_binary)
print(f"Accuracy: {accuracy * 100:.2f}%")
'''