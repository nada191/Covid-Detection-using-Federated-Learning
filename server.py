#from fastapi import FastAPI
import json
from json import JSONDecodeError
#from fastapi.encoders import jsonable_encoder
#from fastapi.responses import JSONResponse
import flwr as fl
import numpy as np
import os
import time
from FLstrategies import *
#from fastapi.middleware.cors import CORSMiddleware




def launch_fl_session(num_rounds: int, ipaddress: str, port: int, resume: bool):
    """
    start flower server and trigger update_strategy event on blockchain
    then connect to clients to perform fl session
    """
    session = int(time.time())
    with open('config_training.json', 'w+') as config_training:
        config = config_training.read()

        try:
            data = json.loads(config)
            data['num_rounds'] = num_rounds
            data['ip_address'] = ipaddress
            data['port'] = port
            data['resume'] = resume
            data['session'] = session
            json.dump(data, config_training)

        except JSONDecodeError:
            data = {}
            data['num_rounds'] = num_rounds
            data['ip_address'] = ipaddress
            data['port'] = port
            data['resume'] = resume
            data['session'] = session
            json.dump(data, config_training)

    # Load last session parameters if they exist
    if not (os.path.exists('./fl_sessions')):
        # create fl_sessions directory if first time
        os.mkdir('fl_sessions')

        # initialise sessions list and initial parameters
    sessions = []
    initial_params = None

    for root, dirs, files in os.walk("./fl_sessions", topdown=False):
        for name in dirs:
            if name.find('Session') != -1:
                sessions.append(name)
    # loop through fl_sessions sub-folders and get the list of directories containing the weights

    if (resume and len(sessions) != 0):
        # test if we will start training from the last session weights and
        # if we have at least a session directory
        if os.path.exists(f'./fl_sessions/{sessions[-1]}/global_session_model.npy'):
            # if the latest session directory contains the global model parameters
            initial_parameters = np.load(f"./fl_sessions/{sessions[-1]}/global_session_model.npy", allow_pickle=True)
            initial_params = initial_parameters[0]
            # load latest session's global model parameters

    strategy_coefs = {"min available clients": 2,
                      "min evaluation clients": 2,
                      "min fitting clients": 2,
                      "fraction of clients for fitting": 1.0,
                      "fraction of clients for evaluation": 1.0}
    # Create strategy and run server
    strategy = SaveModelStrategy(
        fraction_fit=strategy_coefs["fraction of clients for fitting"],
        fraction_eval=strategy_coefs["fraction of clients for evaluation"],
        min_fit_clients=strategy_coefs["min fitting clients"],
        min_eval_clients=strategy_coefs["min evaluation clients"],
        min_available_clients=strategy_coefs["min available clients"],
        initial_parameters=initial_params,
        on_fit_config_fn=get_on_fit_config_fn(),
        on_evaluate_config_fn=evaluate_config,
    )

    fl.server.start_server(
        server_address=ipaddress + ':' + str(port),
        config={"num_rounds": num_rounds},
        grpc_max_message_length=1024 * 1024 * 1024,
        strategy=strategy
    )


launch_fl_session(2, "[::]",  8080, True)
