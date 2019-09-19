import csv
from scipy import ndimage
import numpy as np

lines = []

with open("../../../opt/carnd_p3/data/driving_log.csv") as csvfile:
# with open("../data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[0] != "center":
            lines.append(line)

# * load image and steering in center, left, right
images = []
measurements = []

# create adjusted steering measurements for the side camera images
correction = 0.2 # this is a parameter to tune

for line in lines:
    steering_center = float(line[3])
    for i in range(3):
        source_path = line[i]
        filename = line[i].split('/')[-1]
        current_path = '../../../opt/carnd_p3/data/IMG/' + filename
#         current_path = '../data/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)

        steering_angle = float()
        if i == 0:
            steering_angle = steering_center
        elif i == 1:
            steering_angle = steering_center + correction
        elif i == 2:
            steering_angle = steering_center - correction
        measurements.append(steering_angle)

augumented_images, augumented_measurements = [], []

# * augument data by flipping images and measurements
for image, measurement in zip(images, measurements):
    augumented_images.append(image)
    augumented_measurements.append(measurement)
    augumented_images.append(np.fliplr(image))
    augumented_measurements.append(measurement*(-1))

X_train = np.array(augumented_images)
y_train = np.array(augumented_measurements)

from keras.models import Sequential
# from keras.models import Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
# import matplotlib.pyplot as plt

# * use NVIDIA's end-to-end CNN for self-driving cars
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())
'''
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''
model.save('model.h5')

exit()