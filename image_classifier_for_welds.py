
import tensorflow as tf
import os
import imghdr
import matplotlib
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers, metrics, optimizers, regularizers
from keras.optimizers import Adam, schedules
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import load_model
import shutil
from keras.metrics import Precision, Recall, BinaryAccuracy

#For Colab
from tensorflow.keras.optimizers.schedules import ExponentialDecay

#For VScode
#from keras.optimizers.schedules import ExponentialDecay

data_augmentation = Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
])

resize_and_rescale = Sequential([
  layers.Resizing(180,180),
  layers.Rescaling(1./255)
])

#directory where the data/images are held
data_dir='/content/drive/MyDrive/Colab Notebooks/data_new2'

#Possible extensions of data
image_exts= ['jpeg','jpg','bmp','png']

#Split data into train,validation and test by creating different folders
train_dir = 'train_data'
val_dir = 'val_data'
test_dir = 'test_data'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for image_class in os.listdir(data_dir):
    class_train_dir = os.path.join(train_dir, image_class)
    class_val_dir = os.path.join(val_dir, image_class)
    class_test_dir = os.path.join(test_dir, image_class)
    os.makedirs(class_train_dir, exist_ok=True)
    os.makedirs(class_val_dir, exist_ok=True)
    os.makedirs(class_test_dir, exist_ok=True)

    images = os.listdir(os.path.join(data_dir, image_class))
    train_size = int(len(images) * 0.7)
    val_size = int(len(images) * 0.2)
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    for image in train_images:
      image_path = os.path.join(data_dir, image_class, image)
      new_image_path = os.path.join(class_train_dir, image)
      shutil.copy2(image_path, new_image_path)

    for image in val_images:
      image_path = os.path.join(data_dir, image_class, image)
      new_image_path = os.path.join(class_val_dir, image)
      shutil.copy2(image_path, new_image_path)

    for image in test_images:
      image_path = os.path.join(data_dir, image_class, image)
      new_image_path = os.path.join(class_test_dir, image)
      shutil.copy2(image_path, new_image_path)

train_data = tf.keras.utils.image_dataset_from_directory(train_dir)
val_data = tf.keras.utils.image_dataset_from_directory(val_dir)
test_data = tf.keras.utils.image_dataset_from_directory(test_dir)

train_size = train_data.cardinality().numpy()
val_size = val_data.cardinality().numpy()
test_size = test_data.cardinality().numpy()

train = train_data.take(train_size)
val = val_data.take(val_size)
test = test_data.take(test_size)

#Makes sure that images can be read and have appropirate functions (can be neglected). Only usefull for random data
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path) #reads the file type
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path) #remove file
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)

data=tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator=data.as_numpy_iterator()
batch=data_iterator.next()

#Plot images to see which class bad and good belongs to
fig, ax=plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

#bad is 0 CL
#good is 1 CL


data = data.map(lambda x,y: (x/255, y))
scalled_batch=data.as_numpy_iterator().next()

#Model

base_model = tf.keras.applications.VGG16(
    include_top=False,  # Exclude the classification layers
    weights='imagenet',  # Use pre-trained weights on ImageNet
    input_shape=(256, 256, 3)  # Adjust input shape to match your images
)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

model = tf.keras.Sequential([
    data_augmentation,
    keras.layers.experimental.preprocessing.Resizing(256, 256),  # Resize images to match input shape
    keras.layers.experimental.preprocessing.Rescaling(1./255),  # Rescale pixel values
    base_model,
    layers.GlobalAveragePooling2D(),  # Pooling layer to reduce spatial dimensions
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),  # Add a dense layer for better feature representation
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Exponential Decay Learning Rate (wasn't used)
initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.96

learning_rate_schedule = ExponentialDecay(initial_learning_rate, decay_steps, decay_rate)

#Optimizers
optimizer = tf.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.01)
#optimizer = tf.optimizers.Adam(learning_rate=0.001, weight_decay=0.01)


model.compile(optimizer = optimizer , loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.build(input_shape=(0,256,256,3))
model.summary()

#logs directory
logdir='/content/drive/MyDrive/Colab Notebooks/logs'

#Delete logs -if any- beforre training 
if os.path.exists(logdir):
    # Delete all the files and subdirectories in the log directory
    shutil.rmtree(logdir)

# Recreate the log directory
os.makedirs(logdir)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

hist = model.fit(train, epochs=50, validation_data=val, callbacks=[tensorboard_callback, early_stopping])

#Plot test loss and validation loss
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#Plot test accuracy and validation accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.reset_states()  # Reset Precision metric
    re.reset_states()  # Reset Recall metric
    acc.reset_states()  # Reset BinaryAccuracy metric
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precission:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

#Check that model works with image that wasn't inculded into data
img = cv2.imread('/content/drive/MyDrive/Colab Notebooks/data_test/bad_new/frame0007.jpg')# image directory 
plt.imshow(img)
plt.show()

resize = tf.image.resize(tf.convert_to_tensor(img), (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5:
    print(f'Predicted class is Good')
else:
    print(f'Predicted class is Bad')


#Save model
model.save(os.path.join('/content/drive/MyDrive/Colab Notebooks/models','imageclassifierworking5.h5'))#model path, model name


#Loop in order to check multiple images, if needed
base_directory = '/content/drive/MyDrive/Colab Notebooks/data_test/good_test/'
file_name_format = 'frame{:04d}.jpg'

# Generate the list of image file paths from frame0007.jpg to frame0140.jpg
image_file_paths = [base_directory + file_name_format.format(i) for i in range(1, 141)]

for image_path in image_file_paths:
  img = cv2.imread(image_path)
  #plt.imshow(img)
  #plt.show()

  resize = tf.image.resize(tf.convert_to_tensor(img), (256, 256))
  #plt.imshow(resize.numpy().astype(int))
  #plt.show()

  yhat = model.predict(np.expand_dims(resize/255, 0))
  if yhat > 0.5:
    print(f'Predicted class is Good')
  else:
    print(f'Predicted class is Bad')


#Load model
#new_model = load_model(os.path.join('/content/drive/MyDrive/Colab Notebooks/models','imageclassifierworking5.h5'))