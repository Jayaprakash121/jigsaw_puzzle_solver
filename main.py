import os 
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score

print("Assuming path to image directory is either animal_3_3_jigsaw or human_5_5_jigsaw")
image_dir = input("Enter the path to image directory (Ensure your path ends with '/'): ")
grid_size = int(input("Enter the grid size (3 or 5): "))

if grid_size == 3 :
    ground_truth = pd.read_csv("animal_3_3_jigsaw.csv")
    
    #Remaping the labels
    for row in range(3):
        for col in range(3):
            ground_truth[f'piece_{row}_{col}'] = ground_truth[f'piece_{row}_{col}'].map({
                    '(0, 0)':0, '(0, 1)':1, '(0, 2)':2, 
                    '(1, 0)':3, '(1, 1)':4, '(1, 2)':5, 
                    '(2, 0)':6, '(2, 1)':7, '(2, 2)':8})

elif grid_size == 5 :
    ground_truth = pd.read_csv("human_5_5_jigsaw.csv")
    
    #Remaping the labels
    for row in range(5):
        for col in range(5):
                ground_truth[f'piece_{row}_{col}'] = ground_truth[f'piece_{row}_{col}'].map({
                    '(0, 0)': 0, '(0, 1)': 1, '(0, 2)': 2, '(0, 3)': 3, '(0, 4)': 4,
                    '(1, 0)': 5, '(1, 1)': 6, '(1, 2)': 7, '(1, 3)': 8, '(1, 4)': 9, 
                    '(2, 0)': 10, '(2, 1)': 11, '(2, 2)': 12, '(2, 3)': 13, '(2, 4)': 14, 
                    '(3, 0)': 15, '(3, 1)': 16, '(3, 2)': 17, '(3, 3)': 18, '(3, 4)': 19,
                    '(4, 0)': 20, '(4, 1)': 21, '(4, 2)': 22, '(4, 3)': 23, '(4, 4)': 24})

#Splitting the images into pieces
def split_images(image_name, image_dir, grid_size, count):
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
            count += 1
            pieces.append(piece)
            piece_positions.append(correct_positions[f'piece_{row}_{col}'])
    
    return pieces, np.array(piece_positions), count

#Extracting the features
print("training pieces started")
piece_cnt = 0
training_pieces = []
training_positions = []
for i in range(6000):
    if i%1000 == 0 : print(i)
    image_name = f'{i}.jpg'
    pc, pcp, pc_cnt = split_images(image_name, image_dir, grid_size, piece_cnt)
    piece_cnt = pc_cnt
    training_pieces.append(pc)
    training_positions.append(pcp)

print("Total no. of pieces for training = ", piece_cnt)

training_pieces = np.array(training_pieces)
#print(training_pieces.shape)
training_positions = np.array(training_positions)

training_pieces = training_pieces.reshape(-1, 64, 64, 3)
#print(training_pieces.shape)
training_positions = training_positions.reshape(-1)

# For simplicity, assume the original labels are indices, and we create one-hot encodings
num_samples = piece_cnt
num_classes = grid_size * grid_size

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
    pc, pcp, v = split_images(image_name, image_dir, grid_size, a)
    a = v
    validating_pieces.append(pc)
    validating_positions.append(pcp)
    
print("Total no. of pieces for validating = ", piece_cnt)

validating_pieces = np.array(validating_pieces)
validating_positions = np.array(validating_positions)

validating_pieces = validating_pieces.reshape(-1, 64, 64, 3)
validating_positions = validating_positions.reshape(-1)

# For simplicity, assume the original labels are indices, and we create one-hot encodings
num_samples = piece_cnt
num_classes = grid_size * grid_size

# Convert labels to one-hot encoding
validating_positions_one_hot = np.zeros((num_samples, num_classes))
for i, label in enumerate(validating_positions):
    validating_positions_one_hot[i, label] = 1
    
print("testing pieces started")
piece_cnt = 0
testing_pieces = []
testing_positions = []
for i in range(3000, 6000):
    image_name = f'{i}.jpg'
    pc, pcp, pc_cnt = split_images(image_name, image_dir, grid_size, piece_cnt)
    piece_cnt = pc_cnt
    testing_pieces.append(pc)
    testing_positions.append(pcp)

print("Total no. of pieces for testing = ", piece_cnt)

#Converting the list into array
testing_pieces = np.array(testing_pieces)
testing_positions = np.array(testing_positions)

#Reshaping the array
testing_pieces = testing_pieces.reshape(-1, 64, 64, grid_size)
testing_positions = testing_positions.reshape(-1)  

#Creating a model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(25, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#Training a model
print("training the model")
if grid_size == 3:
    eps = 6
    bs = 250
elif grid_size == 5:
    eps = 30
    bs = 500
    
model = build_model()
history = model.fit(
    training_pieces,  
    training_positions_one_hot,  
    validation_data=(validating_pieces, validating_positions_one_hot),  
    epochs=eps,
    batch_size=bs,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
)

'''
#Importing the model
if grid_size == 3 : from model_3X3 import model
elif grid_size == 5 : from model_5X5 import model
'''

#Predict test pieces
print("Started predicting")
predictions = model.predict(testing_pieces)

threshold = 0.5
predictions_binary = (predictions >= threshold).astype(int)

#Coverting the predictions back to labels because predictions are one-hot encoded
predictions_labels = np.argmax(predictions_binary, axis=1)

#Calculate accuracy
accuracy = accuracy_score(testing_positions, predictions_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")


print("Creating .csv file")
if(grid_size == 3):
    #Remapping the labels
    mapped_predicted_labels = []
    for label in predictions_labels:
        if (label == 0): mapped_predicted_labels.append('(0, 0)')
        if (label == 1): mapped_predicted_labels.append('(0, 1)')
        if (label == 2): mapped_predicted_labels.append('(0, 2)')
        if (label == 3): mapped_predicted_labels.append('(1, 0)')
        if (label == 4): mapped_predicted_labels.append('(1, 1)')
        if (label == 5): mapped_predicted_labels.append('(1, 2)')
        if (label == 6): mapped_predicted_labels.append('(2, 0)')
        if (label == 7): mapped_predicted_labels.append('(2, 1)')
        if (label == 8): mapped_predicted_labels.append('(2, 2)')
    image_name = []
    piece_0_0 = []
    piece_0_1 = []
    piece_0_2 = []
    piece_1_0 = []
    piece_1_1 = []
    piece_1_2 = []
    piece_2_0 = []
    piece_2_1 = []
    piece_2_2 = []
    a = -1
    for i in range(0, 8000*9, 10):
        a += 1
        image_name.append(f'{a}.jpg')
        piece_0_0.append(mapped_predicted_labels[i])
        piece_0_1.append(mapped_predicted_labels[i+1])
        piece_0_2.append(mapped_predicted_labels[i+2])
        piece_1_0.append(mapped_predicted_labels[i+3])
        piece_1_1.append(mapped_predicted_labels[i+4])
        piece_1_2.append(mapped_predicted_labels[i+5])
        piece_2_0.append(mapped_predicted_labels[i+6])
        piece_2_1.append(mapped_predicted_labels[i+7])
        piece_2_2.append(mapped_predicted_labels[i+8])

    data = {
        'image_name': image_name,
        'piece_0_0': piece_0_0, 'piece_0_1': piece_0_1, 'piece_0_2': piece_0_2,
        'piece_1_0': piece_1_0, 'piece_1_1': piece_1_1, 'piece_1_2': piece_1_2,
        'piece_2_0': piece_2_0, 'piece_2_1': piece_2_1, 'piece_2_2': piece_2_2
    }
    
    df_real = pd.DataFrame(data)

    # Save to CSV in append mode
    df_real.to_csv(r'C:\Users\jaipr\PycharmProjects\ml_hacathon\NeuroXTensor ML Hackathon\predicted_animal_3_3_jigsaw.csv', mode='a', index=False)
    
    print("predicted_animal_3_3_jigsaw.csv file created successfully")

elif(grid_size == 5):
    #Remapping the labels
    mapped_predicted_labels = []
    for label in predictions_labels:
        if (label == 0): mapped_predicted_labels.append('(0, 0)')
        if (label == 1): mapped_predicted_labels.append('(0, 1)')
        if (label == 2): mapped_predicted_labels.append('(0, 2)')
        if (label == 3): mapped_predicted_labels.append('(0, 3)')
        if (label == 4): mapped_predicted_labels.append('(0, 4)')
        if (label == 5): mapped_predicted_labels.append('(1, 0)')
        if (label == 6): mapped_predicted_labels.append('(1, 1)')
        if (label == 7): mapped_predicted_labels.append('(1, 2)')
        if (label == 8): mapped_predicted_labels.append('(1, 3)')
        if (label == 9): mapped_predicted_labels.append('(1, 4)')
        if (label == 10): mapped_predicted_labels.append('(2, 0)')
        if (label == 11): mapped_predicted_labels.append('(2, 1)')
        if (label == 12): mapped_predicted_labels.append('(2, 2)')
        if (label == 13): mapped_predicted_labels.append('(2, 3)')
        if (label == 14): mapped_predicted_labels.append('(2, 4)')
        if (label == 15): mapped_predicted_labels.append('(3, 0)')
        if (label == 16): mapped_predicted_labels.append('(3, 1)')
        if (label == 17): mapped_predicted_labels.append('(3, 2)')
        if (label == 18): mapped_predicted_labels.append('(3, 3)')
        if (label == 19): mapped_predicted_labels.append('(3, 4)')
        if (label == 20): mapped_predicted_labels.append('(4, 0)')
        if (label == 21): mapped_predicted_labels.append('(4, 1)')
        if (label == 22): mapped_predicted_labels.append('(4, 2)')
        if (label == 23): mapped_predicted_labels.append('(4, 3)')
        if (label == 24): mapped_predicted_labels.append('(4, 4)')
    image_name = []
    piece_0_0 = []
    piece_0_1 = []
    piece_0_2 = []
    piece_0_3 = []
    piece_0_4 = []
    piece_1_0 = []
    piece_1_1 = []
    piece_1_2 = []
    piece_1_3 = []
    piece_1_4 = []
    piece_2_0 = []
    piece_2_1 = []
    piece_2_2 = []
    piece_2_3 = []
    piece_2_4 = []
    piece_3_0 = []
    piece_3_1 = []
    piece_3_2 = []
    piece_3_3 = []
    piece_3_4 = []
    piece_4_0 = []
    piece_4_1 = []
    piece_4_2 = []
    piece_4_3 = []
    piece_4_4 = []
    a = -1
    for i in range(0, 8000*25, 10):
        a += 1
        image_name.append(f'{a}.jpg')
        piece_0_0.append(mapped_predicted_labels[i])
        piece_0_1.append(mapped_predicted_labels[i+1])
        piece_0_2.append(mapped_predicted_labels[i+2])
        piece_0_3.append(mapped_predicted_labels[i+3])
        piece_0_4.append(mapped_predicted_labels[i+4])
        piece_1_0.append(mapped_predicted_labels[i+5])
        piece_1_1.append(mapped_predicted_labels[i+6])
        piece_1_2.append(mapped_predicted_labels[i+7])
        piece_1_3.append(mapped_predicted_labels[i+8])
        piece_1_4.append(mapped_predicted_labels[i+9])
        piece_2_0.append(mapped_predicted_labels[i+10])
        piece_2_1.append(mapped_predicted_labels[i+11])
        piece_2_2.append(mapped_predicted_labels[i+12])
        piece_2_3.append(mapped_predicted_labels[i+13])
        piece_2_4.append(mapped_predicted_labels[i+14])
        piece_3_0.append(mapped_predicted_labels[i+15])
        piece_3_1.append(mapped_predicted_labels[i+16])
        piece_3_2.append(mapped_predicted_labels[i+17])
        piece_3_3.append(mapped_predicted_labels[i+18])
        piece_3_4.append(mapped_predicted_labels[i+19])
        piece_4_0.append(mapped_predicted_labels[i+20])
        piece_4_1.append(mapped_predicted_labels[i+21])
        piece_4_2.append(mapped_predicted_labels[i+22])
        piece_4_3.append(mapped_predicted_labels[i+23])
        piece_4_4.append(mapped_predicted_labels[i+24])

    data = {
        'image_name': image_name, 
        'piece_0_0': piece_0_0, 'piece_0_1': piece_0_1, 'piece_0_2': piece_0_2, 'piece_0_3': piece_0_3, 'piece_0_4': piece_0_4,
        'piece_1_0': piece_2_0, 'piece_1_1': piece_2_1, 'piece_1_2': piece_2_2, 'piece_1_3': piece_2_3, 'piece_1_4': piece_2_4,
        'piece_2_0': piece_2_0, 'piece_2_1': piece_2_1, 'piece_2_2': piece_2_2, 'piece_2_3': piece_2_3, 'piece_2_4': piece_2_4,
        'piece_3_0': piece_3_0, 'piece_3_1': piece_3_1, 'piece_3_2': piece_3_2, 'piece_3_3': piece_3_3, 'piece_3_4': piece_3_4,
        'piece_4_0': piece_4_0, 'piece_4_1': piece_4_1, 'piece_4_2': piece_4_2, 'piece_4_3': piece_4_3, 'piece_4_4': piece_4_4
    }
    
    df_real = pd.DataFrame(data)

    # Save to CSV in append mode
    df_real.to_csv(r'C:\Users\jaipr\PycharmProjects\ml_hacathon\NeuroXTensor ML Hackathon\predicted_human_5_5_jigsaw.csv', mode='a', index=False)
    
    print("predicted_human_5_5_jigsaw.csv file created successfully")