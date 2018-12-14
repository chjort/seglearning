import argparse
import os
import time
import numpy as np

from keras.models import Model
from keras import losses
from keras import regularizers
from keras.layers import Input, Dense, Flatten, Lambda, GlobalAveragePooling2D, MaxPooling2D, Dropout, Conv2D, Conv2DTranspose, Reshape
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K

def main(argv):

    if argv.local:
        model_path = "models/local_model.h5"
        data_path = argv.data_file
        output_checkpoint_folder = "chkpoints/"
        verbose=1
    else:
        # Define IBM Cloud environment folders.
        input_data_folder = os.environ["DATA_DIR"] + "/"
        output_result_folder = os.environ["RESULT_DIR"] + "/"
        output_checkpoint_folder = os.environ["CHECKPOINT_DIR"] + "/"
        output_log_folder = os.environ["LOG_DIR"] + "/"
        
        # create TensorBoard instance for writing validation metrics
        tb_directory_test = output_log_folder + "logs/tb/test"
        tensorboard_test = TensorBoard(log_dir=tb_directory_test)

        # create TensorBoard instance for writing training metrics
        tb_directory_train = output_log_folder + "logs/tb/train"
        tensorboard_train = TensorBoard(log_dir=tb_directory_train)

        splitter = MetricsSplitter(tensorboard_train,tensorboard_test)
        
        model_path = output_result_folder + "model.h5"
        data_path = input_data_folder + argv.data_file
        verbose=2
    
    print("Loading data... ", end="")
    data = np.load(data_path)
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]
    print("done")

    print("Building model... ")
    input_shape = (argv.shape, argv.shape, 3)
    model = build_unet(input_shape)
    model.summary()

    print("Training model... ")
    epochs = argv.epochs
    batch_size = argv.batch_size
    preprocess_for = argv.preprocess_for
    workers = argv.workers
    patience_stop = argv.patience_stop
    patience_lr = argv.patience_lr
    
    checkpoint = ModelCheckpoint(output_checkpoint_folder + "model-{epoch:02d}-{val_dice_coef:.7f}.h5",
                                     monitor="val_dice_coef", verbose=1, save_best_only=True,
                                     save_weights_only=False, mode="max", period=1)
    early_stop = EarlyStopping(monitor="val_dice_coef", patience=patience_stop, mode="max")
    reduce_lr = ReduceLROnPlateau(monitor="val_dice_coef", factor=0.2, patience=patience_lr, min_lr=1e-7, verbose=1, mode="max")

    callbacks = [checkpoint, early_stop, reduce_lr]
    if not argv.local:
        callbacks.append(splitter)
    
    start_time = time.time()
    model.fit(X_train, Y_train,
              validation_data=(X_val, Y_val),
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              verbose=verbose,
              callbacks=callbacks
             )        
    training_time = time.time() - start_time
    print("Elapsed training time:", training_time)
    
    
    print("Saving model... ", end="")
    model.save(model_path)
    print("done")
    

def iou(y_true, y_pred):
    x11, y11, x12, y12 = y_true[:,0], y_true[:,1], y_true[:,2], y_true[:,3]
    x21, y21, x22, y22 = y_pred[:,0], y_pred[:,1], y_pred[:,2], y_pred[:,3]

    xI1 = K.maximum(x11, K.transpose(x21))
    xI2 = K.minimum(x12, K.transpose(x22))

    yI1 = K.maximum(y11, K.transpose(y21))
    yI2 = K.minimum(y12, K.transpose(y22))

    wI = K.maximum((xI2 - xI1), 0)
    hI = K.maximum((yI2 - yI1), 0)

    inter_area = wI * hI

    y_true_area = (x12 - x11) * (y12 - y11)
    y_pred_area = (x22 - x21) * (y22 - y21)

    union = (y_true_area + K.transpose(y_pred_area)) - inter_area
    
    ious = inter_area / (union + K.epsilon())
    return K.mean(ious)

# Metric function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def build_unet(input_shape):
    input_ = Input(input_shape)
    s = Lambda(lambda x: x / 255) (input_)
    
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[input_], outputs=[outputs])
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])

    return model
        
class MetricsSplitter(Callback):

    def __init__(self, train_tb, test_tb):
        super(MetricsSplitter, self).__init__()
        self.test_tb = test_tb   # TensorBoard callback to handle test metrics
        self.train_tb = train_tb # TensorBoard callback to handle training metrics

    def set_model(self, model):
        self.test_tb.set_model(model)
        self.train_tb.set_model(model)

    def isTestMetric(self,metricName):
        return metricName.find("val")==0 # metrics starting with val are computed on validation/test data

    def on_epoch_end(self, epoch, logs=None):
        # divide metrics up into test and train and route to the appropriate TensorBoard instance
        logs = logs or {}
        train_logs = {}
        test_logs = {}
        for metric in logs.keys():
            if self.isTestMetric(metric):
                test_logs[metric] = logs[metric]
            else:
                train_logs[metric] = logs[metric]
        self.test_tb.on_epoch_end(epoch,test_logs)
        self.train_tb.on_epoch_end(epoch,train_logs)

    def on_train_end(self, x):
        self.test_tb.on_train_end(x)
        self.train_tb.on_train_end(x)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data.csv', help='Directory with data')
    parser.add_argument('--epochs', type=int, default='10', help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default='32', help='Batch size for training and validation')
    parser.add_argument('--shape', type=int, default='224', help='Width and height of input images')
    parser.add_argument('--preprocess_for', type=str, default='vgg16', help='Preprocess for ImageNet model')
    parser.add_argument('--workers', type=int, default='100', help='Number of multiprocessing processes')
    parser.add_argument('--patience_stop', type=int, default='20', help='Epochs without improvement before stopping')
    parser.add_argument('--patience_lr', type=int, default='5', help='Epochs without improvement without reducing learning rate')
    parser.add_argument("--memory", action="store_true", help="Use read data into memroy")
    parser.add_argument("--local", action="store_true", help="Use flag to run locally")
    args = parser.parse_args()
    main(args)