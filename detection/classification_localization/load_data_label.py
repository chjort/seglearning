import numpy as np
from PIL import Image
import os
from zipfile import ZipFile
from pandas.io.parsers import read_csv

from keras.utils import Sequence
from keras.preprocessing import image as k_img
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from keras.applications.inception_v3 import preprocess_input as inception_preprocess

preprocess_dict = {
    "vgg16":vgg16_preprocess,
    "vgg19":vgg19_preprocess,
    "resnet50":resnet50_preprocess,
    "inception":inception_preprocess,
}

def load_dataset(path, test_split=0.2, val_split=0.2):
    data_folder = "/".join(path.split("/")[:-1])
    if path.split(".")[-1] == "zip":
        with ZipFile(path) as z:
            #data_folder = "/".join(zip_path.split("/")[:-1])
            data_file = data_folder + "/data.csv"
            if not os.path.exists(data_file):
                print("Extracting {} to {}".format(path, data_folder))
                z.extractall(data_folder)
    else:
        data_file = path
        
    datalist = read_csv(data_file)
    n_classes = len(np.unique(datalist.values[:,-1]))
    print("n_classes:", n_classes)
    train_list = datalist.iloc[:int(datalist.shape[0]*(1-test_split))]
    test_list = datalist.iloc[int(datalist.shape[0]*(1-test_split)):]
    validation_list = train_list.iloc[int(train_list.shape[0]*(1-val_split)):]
    train_list = train_list.iloc[:int(train_list.shape[0]*(1-val_split))]

    sets = Datasets()
    train_dataset = Dataset(train_list.values, data_folder, n_classes)
    validation_dataset = Dataset(validation_list.values, data_folder, n_classes)
    test_dataset = Dataset(test_list.values, data_folder, n_classes, test=True)
    
    sets.train = train_dataset
    sets.validation = validation_dataset
    sets.test = test_dataset
    
    return sets
        

class Datasets(object):
    def __init__(self):
        self.train = None
        self.validation = None
        self.test = None
        

class Dataset(object):
    
    def __init__(self, datalist, data_folder, n_classes, test=False):
        self._data_folder = data_folder
        if self._data_folder != "/":
            self._data_folder = self._data_folder + "/"
        self._datalist = datalist
        self._n_classes = n_classes
        self._images = None
        self._bboxes = None
        self._labels = None
        self._generator = None
        self._test = test
        
    @property
    def images(self):
        return self._images
    
    @property
    def bboxes(self):
        return self._bboxes
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def generator(self):
        return self._generator
        
    @property
    def N(self):
        return len(self._datalist)
    
    @property
    def n_classes(self):
        return self._n_classes
    
    def set_generator(self, shape, batch_size, onehot=True, preprocess_for=None, shuffle=True):
        self._generator = DataGenerator(self._datalist,
                                        self._data_folder,
                                        shape,
                                        self._n_classes,
                                        batch_size,
                                        onehot,
                                        preprocess_for,
                                        shuffle,
                                        self._test
                                       )
        return self._generator
    
    def load_into_memory(self, shape, onehot=True, preprocess_for=None, shuffle=True):
        N = len(self._datalist)
        h, w = shape[:2]
        X = np.empty((N, h, w, shape[-1]))
        bboxes = np.empty((N, 4))
        labels = np.empty(N)
        
        ridx = np.arange(N)
        if shuffle:
            np.random.shuffle(ridx)
            
        if self._test:
            og_images = []
            og_bboxes = []
            

        # Generate data
        for i, line in enumerate(self._datalist[ridx]):
            imp, imh, imw, x0, y0, x1, y1, label, label_id = line
            # Store image
            im = Image.open(self._data_folder + imp)
            if self._test:
                og_images.append(np.array(im))
                og_bboxes.append([x0,y0,x1,y1])
            im = im.resize((w,h), Image.BICUBIC)
            im = np.array(im)
            if len(im.shape) == 2 or im.shape[-1] == 1:
                im = np.stack((im,)*shape[-1], axis=-1)
            X[i,] = im

            # Store bbox
            bboxes[i] = self._scale_bbox([x0, y0, x1, y1], newshape=shape, oldshape=(imh, imw))
            
            # Store class
            labels[i] = int(label_id)
            
        if preprocess_for is not None:
            preprocess_input = preprocess_dict[preprocess_for]
            X = preprocess_input(X)
        else:
            X = self._normalize(X)

        if onehot:
            labels = np.eye(self._n_classes)[labels.astype(int)]        
        
        if self._test:
            self._images = (X, og_images)
            self._bboxes = bboxes, og_bboxes
            self._lables = labels
            
        else:
            self._images = X
            self._bboxes = bboxes
            self._labels = labels
        
        
    def _scale_bbox(self, bbox, newshape, oldshape):
        old_imgh, old_imgw = oldshape[0], oldshape[1]
        new_imgh, new_imgw = newshape[0], newshape[1] 

        x0, y0, x1, y1 = bbox
        x0 = int(x0 * new_imgw / old_imgw)
        y0 = int(y0 * new_imgh / old_imgh)
        x1 = int(x1 * new_imgw / old_imgw)
        y1 = int(y1 * new_imgh / old_imgh)

        return [x0,y0,x1,y1]
    
    def _normalize(self, images):
        return np.multiply(images.astype(np.float32), 1/255)


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, datalist, data_folder, shape, n_classes, batch_size, onehot=True, preprocess_for=None,  shuffle=True, test=False):
        self._filelist = datalist
        self._n_classes = n_classes
        self._data_folder = data_folder
        self._shape = shape
        self._preprocess_for = preprocess_for
        self.batch_size = batch_size
        self._onehot = onehot
        self.shuffle = shuffle
        self._test = test
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._filelist) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self._indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        filelist_temp = [self._filelist[k] for k in indexes]

        # Generate data
        return self.__data_generation(filelist_temp)

        #return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self._indexes = np.arange(len(self._filelist))
        if self.shuffle == True:
            np.random.shuffle(self._indexes)
            
    def __data_generation(self, filelist_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        h, w = self._shape[:2]
        X = np.empty((self.batch_size, h, w, self._shape[-1]))
        bboxes = np.empty((self.batch_size, 4))
        labels = np.empty(self.batch_size)
        if self._test:
            og_images = []
            og_bboxes = []

        # Generate data
        for i, line in enumerate(filelist_temp):
            imp, imh, imw, x0, y0, x1, y1, label, label_id = line
            # Store image
            im = Image.open(self._data_folder + imp)
            if self._test:
                og_images.append(np.array(im))
                og_bboxes.append([x0, y0, x1, y1])
            im = im.resize((w,h), Image.BICUBIC)
            im = np.array(im)
            if len(im.shape) == 2 or im.shape[-1] == 1:
                im = np.stack((im,)*self._shape[-1], axis=-1)
            X[i,] = im

            # Store bbox
            bboxes[i] = self._scale_bbox([x0, y0, x1, y1], newshape=self._shape, oldshape=(imh, imw))
            
            # Store class
            labels[i] = int(label_id)
            
        if self._preprocess_for is not None:
            preprocess_input = preprocess_dict[self._preprocess_for]
            X = preprocess_input(X)
        else:
            X = self._normalize(X)

        if self._onehot:
            labels = np.eye(self._n_classes)[labels.astype(int)]
        
        if self._test:
            return X, bboxes, og_images, og_bboxes, labels
        else:
            return X, {"box_head":bboxes, "class_head":labels}
    
    def _scale_bbox(self, bbox, newshape, oldshape):
        old_imgh, old_imgw = oldshape[0], oldshape[1]
        new_imgh, new_imgw = newshape[0], newshape[1] 

        x0, y0, x1, y1 = bbox
        x0 = int(x0 * new_imgw / old_imgw)
        y0 = int(y0 * new_imgh / old_imgh)
        x1 = int(x1 * new_imgw / old_imgw)
        y1 = int(y1 * new_imgh / old_imgh)

        return [x0,y0,x1,y1]
    
    def _normalize(self, images):
        return np.multiply(images.astype(np.float32), 1/255)