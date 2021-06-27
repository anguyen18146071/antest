import os
import pandas as pd
#from scipy.misc import imread
from imageio import imread
import math
import numpy as np
import cv2
import keras
from keras import Model
from keras.applications import VGG16
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split


def load_data(input_size = (64,64), data_path =  'GTSRB/Final_Training/Images'):

    pixels = []
    labels = []
    # Loop qua các thư mục trong thư mục Images
    for dir in os.listdir(data_path):
        if dir == '.DS_Store':
            continue

        # Đọc file csv để lấy thông tin về ảnh
        class_dir = os.path.join(data_path, dir)
        info_file = pd.read_csv(os.path.join(class_dir, "GT-" + dir + '.csv'), sep=';')

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

    return pixels, labels

# Đường dẫn ảnh
data_path = 'GTSRB/Final_Training/Images'
pixels, labels = load_data(data_path=data_path)


def split_train_val_test_data(pixels, labels):

    # Chuẩn hoá dữ liệu pixels và labels
    pixels = np.array(pixels)
    labels = keras.utils.np_utils.to_categorical(labels)

    # Nhào trộn dữ liệu ngẫu nhiên
    randomize = np.arange(len(pixels))
    np.random.shuffle(randomize)
    X = pixels[randomize]
    print("X=", X.shape)
    y = labels[randomize]
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25,train_size=0.75)

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = split_train_val_test_data(pixels, labels)


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
epochs = 5
batch_size = 16

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_test,y_test))

model.save("traffic_sign_model.h5")

# Kiểm tra model với dữ liệu mới
print(model.evaluate(X_test, y_test))