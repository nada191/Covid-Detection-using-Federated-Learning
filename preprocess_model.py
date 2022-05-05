from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout, MaxPool2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.layers import Dense, MaxPool2D, Conv2D
import shutil
from pathlib import Path
import splitfolders




def preprocess(data_path,number):#data_path
    # if there is a dataset folder from a previous session, it should be removed
    dirpath = Path('dataset_client_number_'+number)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    # split the data folder into train, test and validation sets
    splitfolders.ratio(data_path, output='dataset_client_number_'+number, seed=42, ratio=(.8, .1, .1), group_prefix=None)
    # Data generator
    train_data_gen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input, zoom_range=0.2, horizontal_flip=True, shear_range=0.2, rescale=1. / 255)
    train = train_data_gen.flow_from_directory(directory='dataset_client_number_'+number+'/train', target_size=(224, 224))
    validation_data_gen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input, rescale=1. / 255)
    valid = validation_data_gen.flow_from_directory(directory='dataset_client_number_'+number+'/val', target_size=(224, 224))
    test_data_gen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input, rescale=1. / 255)
    test = train_data_gen.flow_from_directory(directory='dataset_client_number_'+number+'/test', target_size=(224, 224), shuffle=False)
    return train, test, valid

def vgg_model():

    vgg = VGG16(input_shape=(224, 224, 3), include_top=False)
    for layer in vgg.layers:  # Don't train the parameters again
        layer.trainable = False
    x = Flatten()(vgg.output)
    x = Dense(units=2, activation='sigmoid', name='predictions')(x)

    model = Model(vgg.input, x)
    return model
