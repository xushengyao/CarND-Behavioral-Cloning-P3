import os
import csv
import cv2
import numpy as np
import sklearn
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Convolution2D,Cropping2D,\
Flatten,Lambda
from keras.optimizers import Adam
from keras.models import Model
import matplotlib.pyplot as plt


lines = []
with open('./data/driving_log.csv') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.1)

def randomise_brightness(image):
    augment_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_brightness =0.5+np.random.uniform()
    augment_image[:,:,2] = augment_image[:,:,2]*random_brightness
    augment_image[:,:,2][augment_image[:,:,2]>255]  = 255
    augment_image = cv2.cvtColor(augment_image,cv2.COLOR_HSV2RGB)
    return augment_image

def translation_augment(image, angle, trans_range=10):
    rows,cols = int(image.shape[0]), int(image.shape[1])
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    angle += tr_x / trans_range * 2 * 0.1
    tr_y = 0
    M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    return cv2.warpAffine(image,M,(cols,rows)), angle

def augment_picture(images,angles):
    augmented_images, augmented_angles = [], []
    for image, angle in zip(images,angles):
        augmented_images.append(image)
        augmented_angles.append(angle)
        augmented_images.append(cv2.flip(image,1))
        augmented_angles.append(angle*-1.0)
        #image = randomise_brightness(image)
        image, angle = translation_augment(image, angle)
        augmented_images.append(image)
        augmented_angles.append(angle)
        augmented_images.append(cv2.flip(image,1))
        augmented_angles.append(angle*-1.0)
    return augmented_images, augmented_angles


def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]

            images, angles = [], []
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                correction = 0.3
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                center_name = './data/IMG/'+batch_sample[0].split('/')[-1]
                left_name = './data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = './data/IMG//'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(center_name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)
            # trim image to only see section with road
            augmented_images, augmented_angles = augment_picture(images,angles)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation

#model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.7))
model.add(Dense(100))
model.add(Dropout(0.7))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object  = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=8)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
