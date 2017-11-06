import csv
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import glob
import sklearn
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

#positive and negative images for training (all sized to 64x64)
postiveImagesFolder = "D:/repodata/obj_detect/vehicles"
negativeImagesFolder = "D:/repodata/obj_detect/non-vehicles"
posTrain = glob.glob(postiveImagesFolder + "/**/*.png", recursive=True)
negTrain = glob.glob(negativeImagesFolder + "/**/*.png", recursive=True)

def loadImage (sp) :
    image = cv2.imread(sp)
    normed = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    #cv2.cv2.COLOR_BGR2HSV
    #cv2.cv2.COLOR_BGR2YCrCb
    #normed = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #normed = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    return np.asarray(normed)

test1 = loadImage( "D:/repodata/obj_detect/nonudcars/i8.png")
test2 = loadImage( "D:/repodata/obj_detect/vehicles/udcars/i7.png")

X_pos = [loadImage(l) for l in posTrain]
X_neg = [loadImage(l) for l in negTrain]
y_pos = [1 for i in posTrain]
y_neg = [0 for i in negTrain]

X = X_pos
X.extend(X_neg)
y = y_pos
y.extend(y_neg)
X,y = shuffle (X,y)
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.7)

def generator(X,y, batch_size=32):
    num_samples = len(X)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            imgs = X[offset:offset+batch_size]
            lbls = y[offset:offset+batch_size]
            x_t = np.array(imgs)
            y_t = np.array(lbls)
            yield sklearn.utils.shuffle(x_t, y_t)

batch_size = 32
train_gnrtr = generator(X_train,y_train, batch_size=batch_size)
test_gnrtr = generator(X_test,y_test, batch_size=batch_size)
train_steps = len(X_train) / batch_size
test_steps = len(X_test) / batch_size

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, MaxPool2D
from keras.optimizers import Adam
from keras.regularizers import l1
from keras.constraints import maxnorm
from keras.utils import plot_model

model = Sequential()
model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(64,64,3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
adam = Adam()
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
cp = model.fit_generator(
    train_gnrtr, steps_per_epoch=train_steps, \
    validation_data=test_gnrtr, validation_steps=test_steps, \
    epochs=10)
import cntk as C
import keras.backend as K
C.combine(model.model.outputs).save('detector.bin')
cntk_model = C.load_model('detector.bin')
#model.save('model.h5')
print(cp.history.keys())

import matplotlib.pyplot as plt
plt.plot(cp.history['loss'])
plt.plot(cp.history['val_loss'])
plt.title('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper right')
plt.show()


model.predict(np.array([test1,test2]),batch_size=2)