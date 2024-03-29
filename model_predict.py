

# from load_owndata_bigsqlite import load_own_data,get_total_datanum,get_min_datanum
from load_owndata import load_own_data,get_total_datanum,get_min_datanum,load_own_data_new

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import datetime

import tensorboard
import os
import zipfile
import numpy as np
import tensorflow as tf

import copy

import gc 

from tensorflow import keras
from tensorflow.keras import layers

"""
## Downloading the MosMedData: Chest CT Scans with COVID-19 Related Findings

In this example, we use a subset of the
[MosMedData: Chest CT Scans with COVID-19 Related Findings](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1).
This dataset consists of lung CT scans with COVID-19 related findings, as well as without such findings.

We will be using the associated radiological findings of the CT scans as labels to build
a classifier to predict presence of viral pneumonia.
Hence, the task is a binary classification problem.
"""


# CT_data_path = '/home/n200/D-slot/CT-data/'



# # Download url of normal CT scans.
# url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
# filename = os.path.join(CT_data_path, "CT-0.zip")
# keras.utils.get_file(filename, url)

# # Download url of abnormal CT scans.
# url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
# filename = os.path.join(CT_data_path, "CT-23.zip")
# keras.utils.get_file(filename, url)

# # Make a directory to store the data.
# if not os.path.isdir("/home/n200/D-slot/CT-data/MosMedData"):
#   os.makedirs("/home/n200/D-slot/CT-data/MosMedData")

# # Unzip data in the newly created directory.
# with zipfile.ZipFile("/home/n200/D-slot/CT-data/CT-0.zip", "r") as z_fp:
#     z_fp.extractall("/home/n200/D-slot/CT-data/MosMedData")

# with zipfile.ZipFile("/home/n200/D-slot/CT-data/CT-23.zip", "r") as z_fp:
#     z_fp.extractall("/home/n200/D-slot/CT-data/MosMedData")

"""
## Loading data and preprocessing

The files are provided in Nifti format with the extension .nii. To read the
scans, we use the `nibabel` package.
You can install the package via `pip install nibabel`. CT scans store raw voxel
intensity in Hounsfield units (HU). They range from -1024 to above 2000 in this dataset.
Above 400 are bones with different radiointensity, so this is used as a higher bound. A threshold
between -1000 and 400 is commonly used to normalize CT scans.

To process the data, we do the following:

* We first rotate the volumes by 90 degrees, so the orientation is fixed
* We scale the HU values to be between 0 and 1.
* We resize width, height and depth.

Here we define several helper functions to process the data. These functions
will be used when building training and validation datasets.
"""


import nibabel as nib

from scipy import ndimage


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 8
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]

    # print('current shape',current_depth,current_width,current_height)
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)

    # print('img done')
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


"""
Let's read the paths of the CT scans from the class directories.
"""

# # Folder "CT-0" consist of CT scans having normal lung tissue,
# # no CT-signs of viral pneumonia.
# normal_scan_paths = [
#     os.path.join(CT_data_path, "MosMedData/CT-0", x)
#     for x in os.listdir(os.path.join(CT_data_path,"MosMedData/CT-0"))
# ]
# normal_scan_paths=normal_scan_paths[:10]

# # Folder "CT-23" consist of CT scans having several ground-glass opacifications,
# # involvement of lung parenchyma.
# abnormal_scan_paths = [
#     os.path.join(CT_data_path, "MosMedData/CT-23", x)
#     for x in os.listdir(os.path.join(CT_data_path,"MosMedData/CT-23"))
    
# ]
# abnormal_scan_paths=abnormal_scan_paths[:10]

# print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
# print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))


"""
## Build train and validation datasets
Read the scans from the class directories and assign labels. Downsample the scans to have
shape of 128x128x64. Rescale the raw HU values to the range 0 to 1.
Lastly, split the dataset into train and validation subsets.
"""
data_x=[]
label_y=[]

# each_stage_get_numtotal_datanum=get_total_datanum()




# for i in range(1,8):
#     if i ==1:
#         data_i=load_own_data(i)
#         data_i=np.array([resize_volume(img) for img in data_i])
#         data_x=data_i

#         label = np.array([i for _ in range(len(data_x))])
#         label=tf.one_hot(label,7)
#         label_y=label
#         print("shape label_y:",label_y.shape)
#     else:
#         data_i=load_own_data(i)
#         data_i=np.array([resize_volume(img) for img in data_i])
#         data_x=np.concatenate((data_x,data_i),axis=0)
#         print('data x shape:',data_x.shape)
#         label = np.array([i for _ in range(len(data_i))])
#         label=tf.one_hot(label,7)
#         label_y = np.concatenate((label_y, label), axis=0)
        
#         print("shape label_y:",label_y.shape)

num_index = 0
each_stage_get_num=get_min_datanum()

data_x=np.zeros(shape=[each_stage_get_num*8,128, 128,8])
label_y=np.zeros(shape=[each_stage_get_num*8,8])
for i in range(1,9):

    file_num,data_i=load_own_data_new(i,each_stage_get_num)
    print("file num:",file_num)
    print("num_index:",num_index)
    data_i=np.array([resize_volume(img) for img in data_i])
    data_x[num_index:num_index+file_num]=copy.copy(data_i)
    label = np.array([i for _ in range(len(data_i))])
    data_i=None
    label=tf.one_hot(label,8)
    label_y[num_index:num_index+file_num]=copy.copy(label)
    print("shape label_y:",label_y.shape)
    label=None
    num_index+=file_num

data_x=data_x.astype(np.float32)#import !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
label_y=label_y.astype(np.float32)#import !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# print("data x :",data_x)
print("shape label_y:",label_y.shape)

# # Read and process the scans.
# # Each scan is resized across height, width, and depth and rescaled.

# abnormal_scans=load_own_data(2)

# abnormal_scans = np.array([resize_volume(img) for img in abnormal_scans])
# # print(abnormal_scans)

# normal_scans=load_own_data(7)
# normal_scans = np.array([resize_volume(img) for img in normal_scans])

# # abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
# # normal_scans = np.array([process_scan(path) for path in normal_scan_paths])



# print('abn shape',abnormal_scans.shape)

# # For the CT scans having presence of viral pneumonia
# # assign 1, for the normal ones assign 0.



# abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
# abnormal_labels=tf.one_hot(abnormal_labels,2)
# print('abn labels:',abnormal_labels)
# normal_labels = np.array([0 for _ in range(len(normal_scans))])
# normal_labels=tf.one_hot(normal_labels,2)


# # print('abn label',abnormal_labels.shape)

# # Split data in the ratio 70-30 for training and validation.
# x = np.concatenate((abnormal_scans, normal_scans), axis=0)
# y = np.concatenate((abnormal_labels, normal_labels), axis=0)

x_train, x_val, y_train, y_val  = train_test_split(data_x,label_y, test_size=0.3)
print('x train , x val , y train , y val:',np.array(x_train).shape,np.array(x_val).shape,np.array(y_train).shape,np.array(y_val).shape)
# x_val = np.concatenate((abnormal_scans, normal_scans), axis=0)
# y_val = np.concatenate((abnormal_labels, normal_labels), axis=0)


print("xtrain:",x_train.shape)
print(' ytrain shape',y_train.shape)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

"""
## Data augmentation

The CT scans also augmented by rotating at random angles during training. Since
the data is stored in rank-3 tensors of shape `(samples, height, width, depth)`,
we add a dimension of size 1 at axis 4 to be able to perform 3D convolutions on
the data. The new shape is thus `(samples, height, width, depth, 1)`. There are
different kinds of preprocessing and augmentation techniques out there,
this example shows a few simple ones to get started.
"""

import random

from scipy import ndimage


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        # volume[volume < 0] = 0
        # volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


"""
While defining the train and validation data loader, the training data is passed through
and augmentation function which randomly rotates volume at different angles. Note that both
training and validation data are already rescaled to have values between 0 and 1.
"""


# train_data_chunks = list(np.split(x_train, 3))
# train_labels_chunks = list(np.split(y_train, 3))

# val_data_chunks = list(np.split(x_val, 11))
# val_labels_chunks = list(np.split(y_val, 11))


train_data_chunks = list(np.split(x_train, 4))
train_labels_chunks = list(np.split(y_train, 4))

val_data_chunks = list(np.split(x_val, 4))
val_labels_chunks = list(np.split(y_val, 4))


def genenerator_t():
    for i, j in zip(train_data_chunks, train_labels_chunks):
        for index,data in enumerate(i):

            # print('genenerator_tj.shape',index)
            yield i[index], j[index]

def genenerator_v():
    for i, j in zip(val_data_chunks, val_labels_chunks):
        for index,data in enumerate(i):
            yield i[index], j[index]

# Define data loaders.
# train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
shapes=((128, 128, 8),(8))
shapes1=((128, 128, 8),(8))
train_loader = tf.data.Dataset.from_generator(genenerator_t, (tf.float32, tf.float32),output_shapes=shapes)
validation_loader = tf.data.Dataset.from_generator(genenerator_v, (tf.float32, tf.float32),output_shapes=shapes1)


# train_dataset=iter(train_loader)
# validation_dataset=iter(validation_loader)
# print("RRRRRRRRRRRRRRRRRR",next(train_dataset))
batch_size = 16
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train)).map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)


# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val)).map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# validation_dataset=iter(validation_dataset)
# train_dataset=next(train_dataset)


"""
Visualize an augmented CT scan.
"""

import matplotlib.pyplot as plt

data = train_dataset.take(1)
# print(np.array(list(data)).shape)
# print("data shape:",data.shape)

images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 5]), cmap="gray")


"""
Since a CT scan has many slices, let's visualize a montage of the slices.
"""


def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
print('image shape :',image.shape)
plot_slices(3,2, 128, 128, image[:, :, :6])

"""
## Define a 3D convolutional neural network

To make the model easier to understand, we structure it into blocks.
The architecture of the 3D CNN used in this example
is based on [this paper](https://arxiv.org/abs/2007.13224).
"""


def get_model(width=128, height=128, depth=8):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth,1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu",padding='same')(inputs)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=(2,2,2), padding="same")(x)
    x = layers.Dropout(0.25)(x)


    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu",padding='same')(x)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu",padding='same')(x)
    x = layers.MaxPool3D(pool_size=(2,2,2), padding="same")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu",padding='same')(x)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu",padding='same')(x)
    # x = layers.MaxPool3D(pool_size=(3,3,3), padding="same")(x)

    # x = layers.Conv3D(filters=128, kernel_size=3, activation="relu",padding='same')(x)
    # x = layers.Conv3D(filters=128, kernel_size=3, activation="relu",padding='same')(x)
    # x = layers.MaxPool3D(pool_size=(3,3,3), padding="same")(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=64, kernel_size=3, activation="relu",padding='same')(x)
    # x = layers.MaxPool3D(pool_size=(2,2,2))(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=128, kernel_size=3, activation="relu",padding='same')(x)
    # x = layers.MaxPool3D(pool_size=(2,2,2))(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=256, kernel_size=3, activation="relu",padding='same')(x)
    # # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    # x = layerss.Dropout(0.3)(x)

    outputs = layers.Dense(units=8, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.

model = get_model(width=128, height=128, depth=8)
model.summary()


"""
## Train model
"""

# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=30)

log_dir="./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    # callbacks=[checkpoint_cb, early_stopping_cb],
    callbacks=[checkpoint_cb,tensorboard_callback],
)

"""
It is important to note that the number of samples is very small (only 200) and we don't
specify a random seed. As such, you can expect significant variance in the results. The full dataset
which consists of over 1000 CT scans can be found [here](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1). Using the full
dataset, an accuracy of 83% was achieved. A variability of 6-7% in the classification
performance is observed in both cases.
"""

"""
## Visualizing model performance

Here the model accuracy and loss for the training and the validation sets are plotted.
Since the validation set is class-balanced, accuracy provides an unbiased representation
of the model's performance.
"""

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])


plt.savefig('val_result.jpg')

"""
## Make predictions on a single CT scan
"""

# Load best weights.
model.load_weights("3d_image_classification.h5")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
print('predict shape:',np.expand_dims(x_val[0], axis=0).shape)
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )
