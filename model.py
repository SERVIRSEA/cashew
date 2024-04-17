# Import necessary libraries
import tensorflow as tf
import numpy as np
import os, glob, random
from datetime import datetime
from scipy.ndimage import distance_transform_edt as distance
from tensorflow.keras import layers, models, optimizers, losses, metrics, layers, Model
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, Callback, Callback
from tensorflow.keras.metrics import Precision, Recall
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.optimizers import Adam


# Model and training configuration
BANDS = ['input_image1', 'input_image2', 'input_image3', 'input_image4', 'input_image5']
LABEL = 'class_patch'
FEATURES = BANDS + [LABEL]
TRAIN_SIZE = 7000
EVAL_SIZE = 2000
BATCH_SIZE = 8
EPOCHS = 4
BUFFER_SIZE = 1024
KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]  # Kernel shape for convolutional layers

# Define image shapes for different inputs and class labels
IMAGE_SHAPES = {
    'input_image1': (256, 256, 4),
    'input_image2': (128, 128, 4),
    'input_image3': (64, 64, 6),
    'input_image4': (128, 128, 4),
    'input_image5': (32, 32, 7),
    'class_patch': (256, 256, 1)
}

# Optimizer
optimizer = tf.keras.optimizers.Adam()


# ---------------------
# Data Preparation Functions
# -------


def decode_and_reshape(parsed_features):
    """Decode raw image strings and reshape them according to predefined shapes."""
    for key, shape in IMAGE_SHAPES.items():
        parsed_features[key] = tf.io.decode_raw(parsed_features[key], tf.float32)
        parsed_features[key] = tf.reshape(parsed_features[key], shape)
    return parsed_features

def _parse_function(proto):
    """Parse TFRecord examples and prepare them for the model."""
    # Define the expected structure of the TFRecord file
    keys_to_features = {key: tf.io.FixedLenFeature([], tf.string) for key in IMAGE_SHAPES.keys()}
    
    # Parse the input `tf.Example` proto using the dictionary above
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    
    # Decode and reshape image data
    parsed_features = decode_and_reshape(parsed_features)
    
    # Create inverse labels for binary classification
    labels_inverse = tf.math.abs(parsed_features['class_patch'] - 1)
    labels = tf.concat([labels_inverse, parsed_features['class_patch']], axis=-1)
    
    # Return a dictionary of inputs and the corresponding labels
    inputs = {key: parsed_features[key] for key in IMAGE_SHAPES.keys() if key != 'class_patch'}
    return inputs, labels
    
def _input_function(filenames):
    """
    Prepare a dataset from the TFRecord files.
    
    Args:
        filenames (list): List of file paths to the TFRecord files.
        
    Returns:
        tf.data.Dataset: A TensorFlow dataset ready for training or evaluation.
    """
    # Ensure the dataset is correctly formed and returned
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP', num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# ---------------------
# Model Definition
# ---------------------

class AttentionUNet:
    def __init__(self):
        self.img_shape = (None, None, 4)
        self.df = 32  # Downsampling filters
        self.uf = 32  # Upsampling filters

    def build_unet(self):
        def conv2d(layer_input, filters, dropout_rate=0, bn=False):
            d = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
            if bn:
                d = layers.BatchNormalization()(d)
            d = layers.Activation('relu')(d)
            
            d = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            if bn:
                d = layers.BatchNormalization()(d)
            d = layers.Activation('relu')(d)
            
            if dropout_rate:
                d = layers.Dropout(dropout_rate)(d)
            return d

        def deconv2d(layer_input, filters, bn=False):
            u = layers.UpSampling2D((2, 2))(layer_input)
            u = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(u)
            if bn:
                u = layers.BatchNormalization()(u)
            u = layers.Activation('relu')(u)
            return u

        def attention_block(F_g, F_l, F_int, bn=False):
            g = layers.Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_g)
            if bn:
                g = layers.BatchNormalization()(g)
            x = layers.Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_l)
            if bn:
                x = layers.BatchNormalization()(x)
                
            psi = layers.Add()([g, x])
            psi = layers.Activation('relu')(psi)
            psi = layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(psi)
            
            if bn:
                psi = layers.BatchNormalization()(psi)
            psi = layers.Activation('sigmoid')(psi)
            return layers.Multiply()([F_l, psi])

        # Define model inputs
        input_256 = layers.Input(shape=self.img_shape, name="input_image1")
        input_128 = layers.Input(shape=self.img_shape, name="input_image2")
        input_64 = layers.Input(shape=(None, None, 6), name="input_image3")
        input_128s1 = layers.Input(shape=self.img_shape, name="input_image4")
        input_32 = layers.Input(shape=(None, None, 7), name="input_image5")

        # U-Net architecture
        conv1 = conv2d(input_256, self.df)
        pool1 = layers.MaxPooling2D((2, 2))(conv1)
        pool1 = layers.Concatenate()([pool1, input_128, input_128s1])

        conv2 = conv2d(pool1, self.df * 2, bn=True)
        pool2 = layers.MaxPooling2D((2, 2))(conv2)
        pool2 = layers.Concatenate()([pool2, input_64])

        conv3 = conv2d(pool2, self.df * 4, bn=True)
        pool3 = layers.MaxPooling2D((2, 2))(conv3)
        pool3 = layers.Concatenate()([pool3, input_32])

        conv4 = conv2d(pool3, self.df * 8, dropout_rate=0.5, bn=True)
        pool4 = layers.MaxPooling2D((2, 2))(conv4)

        conv5 = conv2d(pool4, self.df * 16, dropout_rate=0.5, bn=True)

        up6 = deconv2d(conv5, self.uf * 8, bn=True)
        att6 = attention_block(up6, conv4, self.uf * 8, bn=True)
        up6 = layers.Concatenate()([up6, att6])
        conv6 = conv2d(up6, self.uf * 8)

        up7 = deconv2d(conv6, self.uf * 4, bn=True)
        att7 = attention_block(up7, conv3, self.uf * 4, bn=True)
        up7 = layers.Concatenate()([up7, att7])
        conv7 = conv2d(up7, self.uf * 4)

        up8 = deconv2d(conv7, self.uf * 2, bn=True)
        att8 = attention_block(up8, conv2, self.uf * 2, bn=True)
        up8 = layers.Concatenate()([up8, att8])
        conv8 = conv2d(up8, self.uf * 2)

        up9 = deconv2d(conv8, self.uf, bn=True)
        att9 = attention_block(up9, conv1, self.uf, bn=True)
        up9 = layers.Concatenate()([up9, att9])
        conv9 = conv2d(up9, self.uf)

        outputs = layers.Conv2D(2, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(conv9)
        model = Model(inputs=[input_256, input_128, input_128s1, input_64, input_32], outputs=outputs)

        # Freeze layers up to a certain point
        for layer in model.layers:
            layer.trainable = False
            if layer.name == 'activation_9':
                break

        # Print layer names and trainable status for verification
        for layer in model.layers:
            print(layer.name, layer.trainable)

        return model

# ---------------------
# Loss Functions
# ---------------------

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) + 
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
            y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss


# ---------------------
# Metrics and Utility Functions
# ---------------------

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res

def calc_dist_map_batch(y_true):
    y_true_numpy = y_true
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


def f1_score(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def surface_loss_keras(y_true, y_pred):
    
    # Convert tensors to numpy arrays and call the external function.
    y_true_dist_map = tf.numpy_function(func=calc_dist_map_batch, 
                                        inp=[y_true], 
                                        Tout=tf.float32)

    # Make sure the custom op's output has the correct shape.
    y_true_dist_map.set_shape(y_true.get_shape())

    # Calculate the loss value.
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)


# ---------------------
# Training Callbacks
# ---------------------


class AlphaScheduler(Callback):
    def __init__(self, alpha, update_fn):
        super(AlphaScheduler, self).__init__()
        self.alpha = alpha
        self.update_fn = update_fn

    def on_epoch_end(self, epoch, logs=None):
        updated_alpha = self.update_fn(K.get_value(self.alpha))
        K.set_value(self.alpha, updated_alpha)

alpha = K.variable(1, dtype='float32')

def update_alpha(value):
  return np.clip(value - 0.01, 0.01, 1)

def gl_sl_wrapper(alpha):
    def gl_sl(y_true, y_pred):
        return alpha* weighted_bce_dice_loss(y_true, y_pred) +  (1-alpha)* surface_loss_keras(y_true, y_pred)
    return gl_sl


def main():

    # Set paths and dataset parameters
    OUTPUT_DIR = ''
    LOGS_DIR = ''

    # Input all the TFRecord files.
    pattern = '/*.tfrecord.gz'

    # glob the files
    files_list = glob.glob(pattern)

    # Create a dataset from the files.
    dataset = _input_function(files_list)

    # Shuffle the dataset 
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE) 

    # Calculate the total number of items in the dataset.
    total_items_in_dataset = sum(1 for _ in dataset)
    print("total number",total_items_in_dataset)

    # Now, split the dataset into training and testing.
    train_percentage = 0.8  

    num_train = int(total_items_in_dataset * train_percentage)
    num_test = total_items_in_dataset - num_train

    # Split the data (you should calculate the actual sizes of your dataset)
    train_dataset = dataset.take(num_train)
    test_dataset = dataset.skip(num_train)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    weight_path="{}_best_weights.hdf5".format('model')

    checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max', save_weights_only = True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, 
                                       patience=3, 
                                       verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)

    early = EarlyStopping(monitor="val_dice_coef",  mode="max",  patience=6) 
    callbacks_list = [checkpoint, early,AlphaScheduler(alpha, update_alpha)]

    # get the model
    Net=AttentionUNet()
    unet=Net.build_unet()
    print(unet.summary())
    
    # compile the model
    unet.compile(loss=gl_sl_wrapper(alpha),
                 optimizer=Adam(1e-3),
                 metrics=[dice_coef, 'accuracy', Precision(), Recall(),f1_score])

    # fit your model using the datasets
    unet.fit(train_dataset,
             epochs=EPOCHS,
             validation_data=test_dataset,
             shuffle=True,
             callbacks=callbacks_list)


if __name__ == '__main__':
    main()
















