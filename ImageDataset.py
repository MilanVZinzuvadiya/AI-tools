'''
ImageDataset Class

This class generate Image Dataset from given Image Directory
to use into Tensorflow/Keras Environment.

reference: https://www.tensorflow.org/tutorials/load_data/images

'''

import os
import pathlib
import numpy as np
import tensorflow as tf



class ImageDataset:
    __author__ = 'Milan Zinzuvadiya'

    def __init__(self,datadir,batch_size=32,target_img_size=(224,224),Type = 'tensorflow',cache=True,buffer_size=1000):
        self.batch_size = batch_size
        self.target_img_size = target_img_size

        self.data_dir = pathlib.Path(datadir)
        self.image_count = len(list(self.data_dir.glob('*/*.jpg')))
        print('Total images in Dataset: ',self.image_count)

        self.CLASS_NAMES = np.array([item.name for item in self.data_dir.glob('*') if item.is_dir() == True])
        print(self.CLASS_NAMES)

        if Type == 'keras':
            self.kerasGenerate()
        else:
            self.tensorflowGenerate(cache,buffer_size)

    # Generate Image Dataset 
    # Built on keras
    #    
    def kerasGenerate(self):
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_data_gen = image_generator.flow_from_directory(directory=str(self.data_dir),
                                                     batch_size=self.batch_size,
                                                     shuffle=True,
                                                     target_size=self.target_img_size,
                                                     classes = list(self.CLASS_NAMES))
        self.image_generator = train_data_gen

    # Generate Image Dataset 
    # Built on tensorflow
    # 
    def tensorflowGenerate(self,cache,shuffle_buffer_size):
        
        #return label list as bolean values represnted as True for correct class label and False as non correct class label
        def get_label(file_path):
            # convert the path to a list of path components
            parts = tf.strings.split(file_path, os.path.sep)
            # The second to last is the class-directory
            return parts[-2] == self.CLASS_NAMES

        #make image trainable 
        def decode_img(img):
            # convert the compressed string to a 3D uint8 tensor
            img = tf.image.decode_jpeg(img, channels=3)
            # Use `convert_image_dtype` to convert to floats in the [0,1] range.
            img = tf.image.convert_image_dtype(img, tf.float32)
            # resize the image to the desired size.
            return tf.image.resize(img, self.target_img_size)

        #pair trainable image with boolean label list
        def process_path(file_path):
            label = get_label(file_path)
            # load the raw data from the file as a string
            img = tf.io.read_file(file_path)
            img = decode_img(img)
            return img, label
        
        def prepare_for_training(ds, cache, shuffle_buffer_size):
            # This is a small dataset, only load it once, and keep it in memory.
            # use `.cache(filename)` to cache preprocessing work for datasets that don't
            # fit in memory.
            if cache:
                if isinstance(cache, str):
                    ds = ds.cache(cache)
                else:
                    ds = ds.cache()

            ds = ds.shuffle(buffer_size=shuffle_buffer_size)

            # Repeat forever
            ds = ds.repeat()

            ds = ds.batch(self.batch_size)

            # `prefetch` lets the dataset fetch batches in the background while the model
            # is training.
            ds = ds.prefetch(buffer_size=AUTOTUNE)

            return ds

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))
        labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
        self.image_generator = prepare_for_training(labeled_ds,cache,shuffle_buffer_size)

    # get next batch of img,label
    def next(self):
        return next(self.image_generator)
    
    #describe current dataset object
    def __str__(self):
        print("\n{:>20} : {}\n{:>20} : {}\n{:>20} : {}\n".format('Total Images',self.image_count,'Number of Classes',len(self.CLASS_NAMES),'Class Names',self.CLASS_NAMES))
        return " " 