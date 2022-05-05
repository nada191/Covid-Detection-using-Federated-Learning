import flwr as fl
import pandas as pd
from flwr.common import parameters_to_weights
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse

import argparse
import sys
import warnings

    # from fastapi import FastAPI


from preprocess_model import vgg_model, preprocess

if not sys.warnoptions:
    warnings.simplefilter("ignore")
parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--client', action='store', type=int, help='client number')
#parser.add_argument('--adress', action='store', type=str, help='adress of the client')


args = parser.parse_args()
client = vars(args)["client"]

# app = FastAPI()

# @app.post("/participateFL")
# def listen_and_participate (train_start:int, train_end:int, ipadress:str, port:int):

class_type = {0:'Covid',  1 : 'Normal'}
model = vgg_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

data_path = "/Users/macbookair/Desktop/fl-dataset/client"+str(client)+"/"
# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        """Get parameters of the local model."""
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        global train, test, valid
        train, test, valid = preprocess(data_path, str(client))
        hist = model.fit_generator(train, steps_per_epoch=5, epochs=2, validation_data=valid, validation_steps=32)
        results = {
            "loss": hist.history["loss"][0],
            "accuracy": hist.history["accuracy"][0],
            "val_loss": hist.history["val_loss"][0],
            "val_accuracy": hist.history["val_accuracy"][0],
        }
        return model.get_weights(), len(train), results

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy= model.evaluate_generator(test)
        print('loss :', loss, 'accuracy : ', accuracy)
        return float(loss), len(test), {"accuracy": float(accuracy)}


# start Flower client
    # fl.client.start_numpy_client(
    #     server_address=args.ipadress + ":" + str(args.port),
    #     client=FlowerClient(),
    #     grpc_max_message_length=1024 * 1024 * 1024,
    # )
client = FlowerClient()
fl.client.start_numpy_client("[::]:8080", client=client)
