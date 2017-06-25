#!/usr/bin/python3

# Project 3 : Behavioral cloning
# This code sets a model and its parameters to drive a car around a race track. 

# Input: 
# Images (640x480x3 pixels) and corresponding steering angles from a .csv file

# Output:
# The  Keras model is stored in the model.json file and its weights 
# in model.h5

# -----------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------
import csv
import cv2
import numpy as np

import matplotlib.image as mpimg

from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.activations import relu 
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, merge
from keras.models import Model
from keras.models import model_from_json
from keras.optimizers import Adam

import json


# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
fine_tune_mode = True # Collected data are not attached!


np.random.seed(2017) # Maintain coherence among trials

BATCH_SIZE    = 64
NB_EPOCH      = 1 

TRAIN_SHARE   = .80
VALID_SHARE   = .10 

PIP           = 64 # Picture-in-Picture square (side)  
PIP_FROMTOP   = 60 #60
PIP_OFFSET    = 0 # From left and right borders

# FLIP IMAGES and SKIP IF ZERO Area

if fine_tune_mode: 
    flip_ratio = 40 # over 100
    skip_0 = 0 # 0% kept (over 100)
else:
    flip_ratio = 0 #
    skip_0 = 10 # 

my_optimizer = Adam(lr=0.0001)


# -----------------------------------------------------------------------
# Functions collection
# -----------------------------------------------------------------------
def get_csv_file(csv_file):
    ''' 
    Read the csv file containing the train driving log 
    (image file addresses and steering angles)
    Return the entire file as a list.
    Use the skip_zeros function to eliminate some or all images 
    with a zero as steering angle
    '''   
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        driving_log_list = list(reader)
    
    driving_log_list = skip_zeros(driving_log_list)
     
    return driving_log_list


def skip_zeros(full_driving_log, min_abs_angle = 0.0001, hold_percent = skip_0): 
    '''
    Eliminates some or all pictures with steering angle equal to zero
    Input: a list with steereng measure

    Parameters: 
    - min angle to keep (absolute value)
    - percentage (int / 100) to keep
    '''   
    num_frames = len(full_driving_log)
    print(" ")
    print("Num_frames (original)=", num_frames)

    short_driving_log = []
    
    for idx in np.arange(num_frames):
        steer = (np.absolute(float(full_driving_log[idx][3]))) > min_abs_angle
        if steer:
            short_driving_log.append(full_driving_log[idx])
        else:
            if (np.random.randint(100) < hold_percent):
                short_driving_log.append(full_driving_log[idx])

    num_frames = len(short_driving_log)
    print("Num_frames (zeroes removed) =",num_frames)

    return short_driving_log


def split_idx(num_frames, train=.8, valid=.1,shuffle=True):
    ''' 
    Split the original training set indices  
    (idx - is referred to rows in the csv datafile)
    in train, validation and test set. 
    Return indices of the three datasets
    '''  
    frame_set = np.arange(num_frames)
    if shuffle:
        np.random.shuffle(frame_set)
           
    train_max_idx = int(num_frames * train)
    valid_max_idx = int(num_frames * (train+valid))
    
    train_idx = frame_set[np.arange(train_max_idx)]
    valid_idx = frame_set[np.arange(train_max_idx,valid_max_idx)]
    test_idx  = frame_set[np.arange(valid_max_idx,num_frames)]
    
    return train_idx,valid_idx,test_idx


def image_preprocessing (an_image):
    '''
    Preprocess each image. 
    Select the center + the left and right extreme borders of the road 
    excluding the sky.
    Return three smaller images to use as input of the model
    '''   
    # Cut the sky and the near part of the road below the camera
    select_rows = np.arange(PIP_FROMTOP,PIP_FROMTOP+PIP) 

    # Select the center and the borders of the road     
    select_cols_left =   np.arange(PIP_OFFSET, PIP+PIP_OFFSET)
    select_cols_center = np.arange(160+1     -int(PIP/2), 160+1+int(PIP/2))
    select_cols_right =  np.arange(320       -(PIP+PIP_OFFSET), 320-PIP_OFFSET)
    
    new_image_left  =  an_image[select_rows[:, None],select_cols_left,:]
    new_image_center = an_image[select_rows[:, None],select_cols_center,:]
    new_image_right =  an_image[select_rows[:, None],select_cols_right,:]
    
    return new_image_left,new_image_center,new_image_right

       
def batch_iter_from_idx(any_sample_idx, shuffle=True):
    ''' 
    Generates a batch iterator from indices of the csv-rows-indexed dataset.
    Randomly flip the image
    '''
    
    data_size = len(any_sample_idx)
    num_batches_per_epoch = int(data_size/BATCH_SIZE) + 1
        
    # Shuffle the data at each epoch
    if shuffle:
        shuffled_indices = np.random.permutation(np.arange(data_size))
        any_sample_idx   = any_sample_idx[shuffled_indices]
               

    for batch_num in range(num_batches_per_epoch):
        
        # Set the indiges to draw
        start_index = batch_num * BATCH_SIZE
        end_index = min((batch_num + 1) * BATCH_SIZE, data_size) 
        batch_len = end_index-start_index
        
        # Set to zero outputs              
        x_batch_left  = np.zeros((batch_len,PIP,PIP,3), dtype=np.uint8)
        x_batch_center= np.zeros((batch_len,PIP,PIP,3), dtype=np.uint8)
        x_batch_right = np.zeros((batch_len,PIP,PIP,3), dtype=np.uint8)
        y_batch = np.zeros((batch_len,), dtype=np.float)
        
        # Extract images and steering angle
        # In each row r:
        #    [row][0] is the image filename
        #    [row][3] is the steering angle
        i = 0
        
        for row in any_sample_idx[start_index:end_index]:

            y_batch[i]= float(driving_log[row][3])
            
            img_file = driving_log [row] [0] 
            dummy = mpimg.imread(img_file)
 
            if (fine_tune_mode==True)and(np.random.randint(100) < flip_ratio):
                dummy[...,0]= cv2.flip(dummy[...,0], 1) 
                dummy[...,1]= cv2.flip(dummy[...,1], 1) 
                dummy[...,2]= cv2.flip(dummy[...,2], 1)
                y_batch[i] = y_batch[i] * (-1.)

            x_batch_left[i],x_batch_center[i],x_batch_right[i] = image_preprocessing(dummy)
    
            i +=1
            
            yield [x_batch_left,x_batch_center,x_batch_right], y_batch
            

# -----------------------------------------------------------------------
# Read the csv file and prepare the input 
# -----------------------------------------------------------------------
if fine_tune_mode==False:

    # Read in the csv file 
    driving_log = get_csv_file('driving_log.csv')
    num_frames = len(driving_log)

else:

    driving_log   = get_csv_file('/Users/piscab/Desktop/driving_logE.csv')

    driving_dummy = get_csv_file('/Users/piscab/Desktop/driving_log3off.csv')
    driving_log  += driving_dummy
    
    num_frames = len(driving_log)


# Split indeces of training data into a train, validation and test datasets indices.
train_idx, valid_idx, test_idx = split_idx(num_frames,train=TRAIN_SHARE,valid=VALID_SHARE)

# Create geneartors for the three datasets
train_idx_batches = batch_iter_from_idx(train_idx, shuffle=True)
valid_idx_batches = batch_iter_from_idx(valid_idx, shuffle=False)
test_idx_batches  = batch_iter_from_idx(test_idx,  shuffle=False)


# -----------------------------------------------------------------------
# Some other (derived) parameters
# -----------------------------------------------------------------------
samples_per_epoch_param = int(train_idx.shape[0]/BATCH_SIZE)*BATCH_SIZE
nb_val_samples_param    = int(valid_idx.shape[0]/BATCH_SIZE)*BATCH_SIZE
nb_test_samples_param   = int(test_idx.shape [0]/BATCH_SIZE)*BATCH_SIZE

print("Train data set dimension:     ",train_idx.shape)
print("Validation data set dimension:",valid_idx.shape)
print("Test data set dimension:      ",test_idx.shape)
print(" ")


if fine_tune_mode:
# -----------------------------------------------------------------------
# If fine tuning, load the existing model & weights
# -----------------------------------------------------------------------
    with open("model copy.json", 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile(optimizer=my_optimizer, loss='mse')
    weights_file = "model.h5 copy"
    model.load_weights(weights_file)

else:
# -----------------------------------------------------------------------
# Keras model architecture
# -----------------------------------------------------------------------
# Input left center and right image

    img_l  = Input(shape=(PIP,PIP,3))
    img_c  = Input(shape=(PIP,PIP,3))
    img_r  = Input(shape=(PIP,PIP,3))

    img_left  = Lambda(lambda x: x/127.5 -1.,
        input_shape =(PIP,PIP,3), 
        output_shape=(PIP,PIP,3))(img_l)

    img_center= Lambda(lambda x: x/127.5 -1.,
        input_shape =(PIP,PIP,3), 
        output_shape=(PIP,PIP,3))(img_c)

    img_right = Lambda(lambda x: x/127.5 -1.,
        input_shape =(PIP,PIP,3), 
        output_shape=(PIP,PIP,3))(img_r)

# LEFT
    conved_left1  = Convolution2D(32, 5, 5, border_mode='valid',subsample=(2,2))(img_left)  
    relu_left1     = Activation('relu')(conved_left1)
    
    conved_left2   = Convolution2D(32, 5, 5, border_mode='valid',subsample=(2,2))(relu_left1) 
    relu_left2     = Activation('relu')(conved_left2)
    
    conved_left3   = Convolution2D(32, 5, 5, border_mode='valid',subsample=(2,2))(relu_left2) 
    relu_left3     = Activation('relu')(conved_left3)
    
    drop_out_left  = Dropout(0.50)(relu_left3) 
    dense_left     = Dense(16, activation='relu')(drop_out_left)
    flat_left      = Flatten()(dense_left)

# CENTER
    conved_center1   = Convolution2D(32, 5, 5, border_mode='valid',subsample=(2,2))(img_center)  
    relu_center1     = Activation('relu')(conved_center1)
    
    conved_center2   = Convolution2D(32, 5, 5, border_mode='valid',subsample=(2,2))(relu_center1)  
    relu_center2     = Activation('relu')(conved_center2)
    
    conved_center3   = Convolution2D(32, 5, 5, border_mode='valid',subsample=(2,2))(relu_center2)      
    relu_center3     = Activation('relu')(conved_center3)

    drop_out_center = Dropout(0.50)(relu_center3) 
    dense_center    = Dense(16, activation='relu')(drop_out_center)
    flat_center     = Flatten()(dense_center)

# RIGHT
    conved_right1  = Convolution2D(32, 5, 5, border_mode='valid',subsample=(2,2))(img_right) 
    relu_right1    = Activation('relu')(conved_right1)

    conved_right2  = Convolution2D(32, 5, 5, border_mode='valid',subsample=(2,2))(relu_right1) 
    relu_right2    = Activation('relu')(conved_right2)

    conved_right3  = Convolution2D(32, 5, 5, border_mode='valid',subsample=(2,2))(relu_right2) 
    relu_right3    = Activation('relu')(conved_right3)
    
    drop_out_rigth = Dropout(0.50)(relu_right3) 
    dense_right    = Dense(16, activation='relu')(drop_out_rigth)
    flat_right     = Flatten()(dense_right)

#  POOLING
    merge_all  = merge([flat_left,flat_center,flat_right], mode='concat')
    dense_all = Dense(256, activation='relu')(merge_all)
    

# Output layer
    out = Dense(1)(dense_all)

    model = Model([img_l, img_c, img_r], out)




# -----------------------------------------------------------------------
# Train the network using generators for training, validation and test
# -----------------------------------------------------------------------

model.compile(optimizer=my_optimizer, loss='mse')

history = model.fit_generator(train_idx_batches,
    samples_per_epoch= samples_per_epoch_param,
    nb_epoch= NB_EPOCH,
    nb_val_samples= nb_val_samples_param,
    validation_data= valid_idx_batches)

y_test = model.evaluate_generator(test_idx_batches, 
	val_samples=nb_test_samples_param)

print("Memo: test_loss:",y_test)
print(" ")
print(model.summary())


# -----------------------------------------------------------------------
# Save the model and weights
# -----------------------------------------------------------------------
model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(model_json, json_file)

model.save_weights("./model.h5")
print("Saved model to disk")

# -----------------------------------------------------------------------
# THE END 
# -----------------------------------------------------------------------
