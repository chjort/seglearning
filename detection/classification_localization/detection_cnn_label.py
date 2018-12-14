from load_data_label import load_dataset
import argparse
import os
import time
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras import losses
from keras import regularizers
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Conv2D, Reshape
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
    data = load_dataset(data_path)
    n_classes = data.train.n_classes
    print("done")

    print("Building model... ")
    input_shape = (argv.shape, argv.shape, 3)
    #n_classes = argv.n_classes
    model = build_clfbbox_model(input_shape, n_classes)
    model.summary()

    print("Training model... ")
    epochs = argv.epochs
    batch_size = argv.batch_size
    preprocess_for = argv.preprocess_for
    workers = argv.workers
    patience_stop = argv.patience_stop
    patience_lr = argv.patience_lr
    
    checkpoint = ModelCheckpoint(output_checkpoint_folder + "model-{epoch:02d}-{val_box_head_iou:.7f}.h5",
                                     monitor="val_box_head_iou", verbose=1, save_best_only=True,
                                     save_weights_only=False, mode="max", period=1)
    early_stop = EarlyStopping(monitor="val_box_head_iou", patience=patience_stop, mode="max")
    reduce_lr = ReduceLROnPlateau(monitor="val_box_head_iou", factor=0.2, patience=patience_lr, min_lr=1e-7, verbose=1, mode="max")

    callbacks = [checkpoint, early_stop, reduce_lr]
    if not argv.local:
        callbacks.append(splitter)
    
    if not argv.memory:
        print("Training using generator")
        traingen = data.train.set_generator(shape=input_shape,
                                            batch_size=batch_size,
                                            preprocess_for=preprocess_for,
                                            onehot=True
                                           )
        
        valgen = data.validation.set_generator(shape=input_shape,
                                              batch_size=batch_size,
                                              preprocess_for=preprocess_for,
                                              onehot=True
                                              )
        
        start_time = time.time()
        model.fit_generator(generator=traingen,
                            validation_data=valgen,
                            epochs=epochs,
                            use_multiprocessing=True,
                            workers=workers,
                            shuffle=True,
                            verbose=verbose,
                            callbacks=callbacks
                           )
    else:
        print("Loading data into memory... ", end="")
        data.train.load_into_memory(shape=input_shape, preprocess_for=preprocess_for, onehot=True)
        data.validation.load_into_memory(shape=input_shape, preprocess_for=preprocess_for, onehot=True)
        print("done")
        
        start_time = time.time()
        #model.fit(data.train.images, data.train.bboxes,
        #          validation_data=(data.validation.images, data.validation.bboxes),
        #          epochs=epochs,
        #          batch_size=batch_size,
        #          shuffle=True,
        #          verbose=verbose,
        #          callbacks=callbacks
        #         )
        
        # Class + box
        model.fit(x=data.train.images,
                  y={"class_head":data.train.labels, "box_head": data.train.bboxes},
                  validation_data=(data.validation.images,
                                   {"class_head":data.validation.labels, "box_head": data.validation.bboxes}),
                  epochs=epochs,
                  batch_size=49,
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

def build_clfbbox_model(input_shape, n_classes):
    detection_net = VGG16(input_shape=input_shape, weights="imagenet", include_top=False)
    for layer in detection_net.layers:
        layer.trainable = False
    size1, size2 = detection_net.output.shape[1].value, detection_net.output.shape[2].value
    
    #boxhead = Conv2D(filters=16, kernel_size=(size1,size2), activation="relu", name="box_conv2d1")(detection_net.output)
    boxhead = Conv2D(filters=4, kernel_size=(size1,size2), kernel_regularizer=regularizers.l2(0.01), name="box_conv2d2")(detection_net.output)#(boxhead)
    boxhead = Reshape((4,), name="box_head")(boxhead)
    
    #classhead = Conv2D(filters=8, kernel_size=(size1,size2), activation="relu", name="class_conv2d1")(detection_net.output)
    classhead = Conv2D(filters=n_classes, kernel_size=(size1,size2), activation="softmax", name="class_conv2d2")(detection_net.output)#(classhead)
    classhead = Reshape((n_classes,), name="class_head")(classhead)

    model = Model(inputs=detection_net.input, outputs=[boxhead, classhead])

    optimizer = Adam(lr=0.001)
    
    head_losses = {"box_head":"mean_squared_error", "class_head":"categorical_crossentropy"}
    head_metrics = {"box_head":iou, "class_head":"accuracy"}
    loss_weights = {"class_head":1, "box_head":1}

    model.compile(optimizer=optimizer,
                  loss=head_losses,
                  loss_weights={"class_head":1, "box_head":0.9},
                  metrics=head_metrics)
    
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
    #parser.add_argument('--n_classes', type=int, default='2', help='Number of class labels')
    parser.add_argument('--preprocess_for', type=str, default='vgg16', help='Preprocess for ImageNet model')
    parser.add_argument('--workers', type=int, default='100', help='Number of multiprocessing processes')
    parser.add_argument('--patience_stop', type=int, default='20', help='Epochs without improvement before stopping')
    parser.add_argument('--patience_lr', type=int, default='5', help='Epochs without improvement without reducing learning rate')
    parser.add_argument("--memory", action="store_true", help="Use read data into memroy")
    parser.add_argument("--local", action="store_true", help="Use flag to run locally")
    args = parser.parse_args()
    main(args)