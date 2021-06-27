import os
import pickle

import cv2
import keras.utils.np_utils
import numpy as np
import pandas as pd
import pixel as pixel
from cv2 import imread
from keras import Input, Model
from keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split



def load_data(input_size=(64,64), data_path= 'GTSRB/Final_Training/Images'):
    pixels =[]
    labels =[]
    # loop qua cac thu muc trong thu muc Image
    for dir in os.listdir(data_path):
        if dir == '.DS_Store':
            continue
        # doc file csv de lay thong tin va anh
        class_dir = os.path.join(data_path, dir)
        info_file = pd.read_csv(os.path.join(class_dir,"GT-" +dir+'.csv'),sep=';')
        # Lăp trong file
        for row in info_file.iterrows():
            # Đọc ảnh
            pixel = imread(os.path.join(class_dir, row[1].Filename))
            # Trích phần ROI theo thông tin trong file csv
            pixel = pixel[row[1]['Roi.X1']:row[1]['Roi.X2'], row[1]['Roi.Y1']:row[1]['Roi.Y2'], :]
            # Resize về kích cỡ chuẩn
            img = cv2.resize(pixel, input_size)

            # Thêm vào list dữ liệu
            pixels.append(img)

            # Thêm nhãn cho ảnh
            labels.append(row[1].ClassId)
            # Chuẩn hoá dữ liệu pixels và labels
            pixels = np.array(pixels)
            labels = keras.utils.np_utils.to_categorical(labels)
            file = open('pix.data', 'wb')
            # dump information to that file
            pickle.dump((pixels, labels), file)
            # close the file
            file.close()

            return
def save_data():
    file = open('pix.data', 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)

    # close the file
    file.close()

    print(pixels.shape)
    print(labels.shape)

    return pixels, labels

load_data()
X,y = save_data()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25,train_size=0.75)

print(X_train.shape)
print(y_train.shape)

#  tao model
def built_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False
        # Tao model
    input = Input(shape=(64, 64, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)
    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)  # dàn phẳng các bức ảnh thành mảng 1 chiều
    x = Dense(4096, activation='relu', name='fc1')(x)  # lớp dense
    x = Dropout(0.5)(x)  # giảm việc bị overfiting
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc3')(x)
    x = Dropout(0.5)(x)
    x = Dense(43, activation='softmax', name='predictions')(x)
    my_model = Model(inputs=input, outputs=x)

    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return my_model
model = built_model()
# Train model
epochs = 10
batch_size = 16

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_test,y_test))

model.save("traffic_sign_model.h5")