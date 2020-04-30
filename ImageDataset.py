'''
ImageDataset Class

This class generate Image Dataset from given Image Directory
to use into Tensorflow/Keras Environment.


'''

import os
import pathlib
import numpy as np
import tensorflow as tf



class ImageDataset:
    __author__ = 'Milan Zinzuvadiya'

    def __init__(self,datadir,batch_size,target_img_size,gen_type = 'keras'):
        self.batch_size = batch_size
        self.target_img_size = target_img_size

        self.data_dir = pathlib.Path(datadir)
        self.image_count = len(list(self.data_dir.glob('*/*.jpg')))
        print('Total images in Dataset: ',self.image_count)

        self.CLASS_NAMES = np.array([item.name for item in self.data_dir.glob('*') if item.is_dir() == True])
        print(self.CLASS_NAMES)

        if gen_type == 'keras':
            self.kerasGenerate()
    

    def kerasGenerate(self):
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_data_gen = image_generator.flow_from_directory(directory=str(self.data_dir),
                                                     batch_size=self.batch_size,
                                                     shuffle=True,
                                                     target_size=self.target_img_size,
                                                     classes = list(self.CLASS_NAMES))
        self.image_generator = train_data_gen

    def next(self):
        return next(self.image_generator)
    
    def __str__(self):
        print("\n{:>20} : {}\n{:>20} : {}\n{:>20} : {}\n".format('Total Images',self.image_count,'Number of Classes',len(self.CLASS_NAMES),'Class Names',self.CLASS_NAMES))
        return " " 