import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
import os
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.regularizers import l2
from keras.optimizers import RMSprop
import keras
# Set your image directory
# Use double backslashes to escape the backslash
image_directory = r'D:/winter class Hands on/archive/cell_images'

np.random.seed(1000)
SIZE = 150

# Create an instance of ImageDataGenerator for data augmentation
# convert in the same resolution 
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)



# Load and preprocess the dataset using the generator
dataset = []
label = []

parasitized_images = os.listdir(image_directory + '/Parasitized/')
for i, image_name in enumerate(parasitized_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + '/Parasitized/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)

uninfected_images = os.listdir(image_directory + '/Uninfected/')
for i, image_name in enumerate(uninfected_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + '/Uninfected/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

import matplotlib.pyplot as plt

# Count the number of samples in each class
unique, counts = np.unique(label, return_counts=True)
class_counts = dict(zip(unique, counts))




from keras.utils import to_categorical
# Split the dataset
#X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, random_state=0)

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Assuming you have dataset and label

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size=0.20, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create an ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Set the batch size for data augmentation
batch_size = 32

# Create augmented data generator for training
augmented_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

# Batch processing to normalize data
batch_size = 32
for i in range(0, len(X_train), batch_size):
    X_train[i:i+batch_size] = normalize(X_train[i:i+batch_size], axis=1)
    print("Batch " + str(i))

# Similarly, normalize X_test in batches
for i in range(0, len(X_test), batch_size):
    X_test[i:i+batch_size] = normalize(X_test[i:i+batch_size], axis=1)
    print("Batch " + str(i))

# Similarly, normalize X_test in batches
for i in range(0, len(y_test), batch_size):
    X_test[i:i+batch_size] = normalize(X_test[i:i+batch_size], axis=1)
    print("Batch " + str(i))


# Similarly, normalize X_test in batches
for i in range(0, len(y_train), batch_size):
    X_test[i:i+batch_size] = normalize(X_test[i:i+batch_size], axis=1)
    print("Batch " + str(i))


# Similarly, normalize X_test in batches
for i in range(0, len(y_val), batch_size):
    X_test[i:i+batch_size] = normalize(X_test[i:i+batch_size], axis=1)
    print("Batch " + str(i))

# Similarly, normalize X_test in batches
for i in range(0, len(X_val), batch_size):
    X_test[i:i+batch_size] = normalize(X_test[i:i+batch_size], axis=1)
    print("Batch " + str(i))



# ### Build the model

#############################################################
###2 conv and pool layers. with some normalization and drops in between.
    
print("Model part")

INPUT_SHAPE = (SIZE, SIZE, 3)   #change to (SIZE, SIZE, 3)
inp = keras.layers.Input(shape=INPUT_SHAPE)

conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
norm1 = keras.layers.BatchNormalization(axis = -1)(pool1)
drop1 = keras.layers.Dropout(rate=0.2)(norm1)
conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
norm2 = keras.layers.BatchNormalization(axis = -1)(pool2)
drop2 = keras.layers.Dropout(rate=0.2)(norm2)

flat = keras.layers.Flatten()(drop2)  #Flatten the matrix to get it ready for dense.

hidden1 = keras.layers.Dense(512, activation='relu')(flat)
norm3 = keras.layers.BatchNormalization(axis = -1)(hidden1)
drop3 = keras.layers.Dropout(rate=0.2)(norm3)
hidden2 = keras.layers.Dense(256, activation='relu')(drop3)

norm4 = keras.layers.BatchNormalization(axis = -1)(hidden2)
drop4 = keras.layers.Dropout(rate=0.2)(norm4)

out = keras.layers.Dense(2, activation='softmax')(drop4)

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])




print(model.summary())

# Train the model using the generator
batch_size = 32
epochs = 20


history = model.fit(
    augmented_generator,
    steps_per_epoch=len(X_train)//batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    verbose=1
)





print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_val), np.array(y_val))[1]*100))
# Save the model
model.save('malaria_model_10epochs_with_batches_and_augmentation.h5')

# Plot training and validation metrics
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
