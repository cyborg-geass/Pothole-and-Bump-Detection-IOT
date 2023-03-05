import pandas as pd
import numpy as np

features = pd.read_csv(r"Dataset.csv")

features.head(10)
features.isna().sum()
features = features.drop('Unnamed: 4', axis = 1)
features.head(10)
features.isna().sum()

all_image_names = features['Image ID']
features = features.drop('Image ID', axis = 1)

pothole_or_not = features['Pothole']
pothole_info = features['Number of Potholes']
pothole_level = features['Level']

pothole_or_not = pd.get_dummies(pothole_or_not, columns = ['Pothole'])
pothole_or_not.columns = ['Normal Road', 'Pothole']
pothole_level = pd.get_dummies(pothole_level, columns = ['Level'])

features = pothole_or_not.join(pothole_info)
features = features.join(pothole_level)

print(features.head(10))

from sklearn.utils import shuffle

image_names_shuffled, labels_shuffled = shuffle(all_image_names, features)
from sklearn.model_selection import train_test_split

X_train_image_names, X_test_image_names, y_train, y_test = train_test_split(
    image_names_shuffled, labels_shuffled, test_size=0.3, random_state=1)
type(y_train)

train_pothole_or_not = y_train[['Normal Road', 'Pothole']]
train_pothole_info = y_train['Number of Potholes']
train_pothole_level = y_train[['A', 'B', 'C', 'S']]

test_pothole_or_not = y_test[['Normal Road', 'Pothole']]
test_pothole_info = y_test['Number of Potholes']
test_pothole_level = y_test[['A', 'B', 'C', 'S']]
y_train.shape

train_pothole_or_not = np.array(train_pothole_or_not)
train_pothole_info = np.array(train_pothole_info)
train_pothole_level = np.array(train_pothole_level)

test_pothole_or_not = np.array(test_pothole_or_not)
test_pothole_info = np.array(test_pothole_info)
test_pothole_level = np.array(test_pothole_level)

import cv2

def get_image(file_location):
    img = cv2.imread(file_location)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img/255
    img = cv2.resize(img,(200, 200))
    return img
import keras
import numpy as np

class My_Custom_Generator(keras.utils.Sequence) :
  
  def __init__(self, image_filenames, pothole, pothole_number, pothole_level, batch_size = 128) :
    self.image_filenames = image_filenames
    self.pothole = pothole
    self.pothole_number = pothole_number
    self.pothole_level = pothole_level
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return int((np.ceil(len(self.image_filenames) / float(self.batch_size))))
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    a = self.pothole[idx * self.batch_size : (idx+1) * self.batch_size]
    # b = self.pothole_number[idx * self.batch_size : (idx+1) * self.batch_size]
    c = self.pothole_level[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = [np.array(a), np.array(c)] #, np.array(c)]

    return np.array([
            get_image(f'Unified Dataset/{str(file_name)}.jpg')
               for file_name in batch_x]), batch_y
batch_size = 60

my_training_batch_generator = My_Custom_Generator(X_train_image_names, train_pothole_or_not, train_pothole_info, train_pothole_level, batch_size)
my_validation_batch_generator = My_Custom_Generator(X_test_image_names, test_pothole_or_not, test_pothole_info, test_pothole_level, batch_size)
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input, Activation, Add, BatchNormalization

input_ = Input(shape = (200, 200, 1), name = "Input Layer")

conv_1 = Conv2D(32, kernel_size = (4, 4), name = "conv_1")(input_)
act_1 = Activation("relu", name = "act_1")(conv_1)
pool_1 = MaxPooling2D(pool_size = (8, 8), strides = (1, 1), padding = "valid", name = "pool_1")(act_1)

conv_2 = Conv2D(64, kernel_size = (8, 8), name = "conv_2")(pool_1)
act_2 = Activation("relu", name = "act_2")(conv_2)
pool_2 = MaxPooling2D(pool_size = (4, 4), strides = (1, 1), padding = "valid", name = "pool_2")(act_2)

conv_3 = Conv2D(32, kernel_size = (4, 4), name = "conv_3")(pool_2)
act_3 = Activation("relu", name = "act_3")(conv_3)
pool_3 = MaxPooling2D(pool_size = (8, 8), strides = (1, 1), padding = "valid", name = "pool_3")(act_3)

flat_1 = Flatten(name = "flat_1")(pool_3)

dense_1 = Dense(128, activation = "relu", name = "dense_1")(flat_1)
batch_1 = BatchNormalization(name = "batch_1")(dense_1)
dense_2 = Dense(64, activation = "relu", name = "dense_2")(batch_1)
batch_2 = BatchNormalization(name = "batch_2")(dense_2)
dense_3 = Dense(32, activation = "relu", name = "dense_3")(batch_2)
isPothole = Dense(2, activation = "softmax", name = "pothole")(dense_3)

conv_4 = Conv2D(16, kernel_size = (4, 4), name = "conv_4")(pool_3)
act_4 = Activation("relu", name = "act_4")(conv_4)
pool_4 = MaxPooling2D(pool_size = (4, 4), strides = (1, 1), padding = "valid", name = "pool_4")(act_4)

# flat_2 = Flatten(name = "flat_2")(pool_4)

# dense_4 = Dense(128, activation = "relu", name = "dense_4")(flat_2)
# batch_3 = BatchNormalization(name = "batch_3")(dense_4)
# add_1 = Add(name = "add_1")([batch_1, batch_3])

# dense_5 = Dense(64, activation = "relu", name = "dense_5")(add_1)
# batch_4 = BatchNormalization(name = "batch_4")(dense_5)
# pothole_num = Dense(1, activation = "relu", name = "pothole_number")(batch_2)

conv_5 = Conv2D(16, kernel_size = (8, 8), name = "conv_5")(pool_4)
act_5 = Activation("relu", name = "act_5")(conv_5)
pool_5 = MaxPooling2D(pool_size = (4, 4), strides = (1, 1), padding = "valid", name = "pool_5")(act_5)

flat_2 = Flatten(name = "flat_2")(pool_5)

dense_4 = Dense(128, activation = "relu", name = "dense_4")(flat_2)
batch_5 = BatchNormalization(name = "batch_5")(dense_4)
add_2 = Add(name = "add_2")([batch_1, batch_5])
batch_6 = BatchNormalization(name = "batch_6")(add_2)
dense_5 = Dense(64, activation = "relu", name = "dense_5")(batch_6)
batch_7 = BatchNormalization(name = "batch_7")(dense_5)
# add_3 = Add(name = "add_3")([batch_7, batch_2])
dense_6 = Dense(32, activation = "relu", name = "dense_6")(batch_7)
drop_2 = Dropout(rate = 0.4, name = "drop_2")(dense_6)
dense_7 = Dense(16, activation = "relu", name = "dense_7")(drop_2)
drop_3 = Dropout(rate = 0.4, name = "drop_3")(dense_7)
dense_8 = Dense(8, activation = "relu", name = "dense_8")(drop_3)
pothole_level = Dense(4, activation = "softmax", name = "pothole_level")(dense_8)

model = Model(inputs = input_, outputs = [isPothole, pothole_level])

opt = tf.keras.optimizers.SGD(learning_rate = 0.01)

model.compile(
              loss = {
                  'pothole' : 'binary_crossentropy',
                #   'pothole_number' : 'categorical_crossentropy'
                  'pothole_level' : 'categorical_crossentropy'
              },
              optimizer = opt,
              metrics = ['accuracy']
)

model.summary()

# keras.utils.plot_model(model, "model.png", show_shapes = True, show_layer_names = True)

# class Logger(keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     pothole_accuracy = logs.get('pothole_accuracy')
#     pothole_number_accuracy = logs.get('pothole_number_accuracy')
#     pothole_level_accuracy = logs.get('pothole_level_accuracy')

#     val_pothole_accuracy = logs.get('val_pothole_accuracy')
#     val_pothole_number_accuracy = logs.get('val_pothole_number_accuracy')
#     val_pothole_level_accuracy = logs.get('val_pothole_level_accuracy')

#     print('='*30, epoch + 1, '='*30)
#     print(f'pothole_accuracy: {pothole_accuracy:.2f}, pothole_number_accuracy: {pothole_number_accuracy:.2f}, pothole_level_accuracy: {pothole_level_accuracy:.2f}')
#     print(f'val_pothole_accuracy: {val_pothole_accuracy:.2f}, val_pothole_number_accuracy: {val_pothole_number_accuracy:.2f}, val_pothole_level_accuracy:{val_pothole_level_accuracy:.2f}')
_ = model.fit_generator (generator=my_training_batch_generator,
                   steps_per_epoch = int(21000 // batch_size),
                   epochs = 100,
                #    verbose = False,
                   validation_data = my_validation_batch_generator,
                #    callbacks = [
                #                 Logger(),
                #                 keras.callbacks.TensorBoard(log_dir = './logs')
                #                 ],
                   validation_steps = int(9000 // batch_size))
print(model.predict("saved_img.jpg"))

if model.predict("saved_img.jpg")['Pothole']==1:
  from machine import Pin, UART, I2C
  from ssd1306 import SSD1306_I2C

  import utime, time

  i2c=I2C(0,sda=Pin(0), scl=Pin(1), freq=400000)
  oled = SSD1306_I2C(128, 64, i2c)

  gpsModule = UART(1, baudrate=9600, tx=Pin(4), rx=Pin(5))
  print(gpsModule)

  buff = bytearray(255)

  TIMEOUT = False
  FIX_STATUS = False

  latitude = ""
  longitude = ""
  satellites = ""
  GPStime = ""

  def getGPS(gpsModule):
      global FIX_STATUS, TIMEOUT, latitude, longitude, satellites, GPStime
      
      timeout = time.time() + 8 
      while True:
          gpsModule.readline()
          buff = str(gpsModule.readline())
          parts = buff.split(',')
      
          if (parts[0] == "b'$GPGGA" and len(parts) == 15):
              if(parts[1] and parts[2] and parts[3] and parts[4] and parts[5] and parts[6] and parts[7]):
                  print(buff)
                  
                  latitude = convertToDegree(parts[2])
                  if (parts[3] == 'S'):
                      latitude = -latitude
                  longitude = convertToDegree(parts[4])
                  if (parts[5] == 'W'):
                      longitude = -longitude
                  satellites = parts[7]
                  GPStime = parts[1][0:2] + ":" + parts[1][2:4] + ":" + parts[1][4:6]
                  FIX_STATUS = True
                  break
                  
          if (time.time() > timeout):
              TIMEOUT = True
              break
          utime.sleep_ms(500)
          
  def convertToDegree(RawDegrees):

      RawAsFloat = float(RawDegrees)
      firstdigits = int(RawAsFloat/100) 
      nexttwodigits = RawAsFloat - float(firstdigits*100) 
      
      Converted = float(firstdigits + nexttwodigits/60.0)
      Converted = '{0:.6f}'.format(Converted) 
      return str(Converted)
      
      
  while True:
      
      getGPS(gpsModule)

      if(FIX_STATUS == True):
          print("Printing GPS data...")
          print(" ")
          print("Latitude: "+latitude)
          print("Longitude: "+longitude)
          print("Satellites: " +satellites)
          print("Time: "+GPStime)
          print("----------------------")
          
          oled.fill(0)
          oled.text("Lat: "+latitude, 0, 0)
          oled.text("Lng: "+longitude, 0, 10)
          oled.text("Satellites: "+satellites, 0, 20)
          oled.text("Time: "+GPStime, 0, 30)
          oled.show()
          
          FIX_STATUS = False
          
      if(TIMEOUT == True):
          print("No GPS data is found.")
          TIMEOUT = False