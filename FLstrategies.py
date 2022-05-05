import flwr as fl
import numpy as np
import os
from typing import Callable, Dict, Optional, Tuple
from flwr.common import Parameters, Scalar, Weights
from typing import Callable, Dict
import datetime as dt
import json

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self,
                 *,
                 fraction_fit: float = 0.1,
                 fraction_eval: float = 0.1,
                 min_fit_clients: int = 2,
                 min_eval_clients: int = 2,
                 min_available_clients: int = 2,
                 eval_fn: Optional[
                     Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
                 ] = None,
                 on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
                 on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
                 accept_failures: bool = True,
                 initial_parameters: Parameters
                 ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )

        self.contribution = {}

    def aggregate_fit(
            self,
            rnd,
            results,
            failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:

            # get num_rounds from config_training json file to be use to verify
            # if the current round is the first round
            with open('config_training.json', 'r') as config_training:
                config = config_training.read()
                data = json.loads(config)
                num_rounds = data['num_rounds']
                session = data['session']

            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")

            if not os.path.exists(f"./fl_sessions/Session-{session}"):
                os.makedirs(f"./fl_sessions/Session-{session}")
                if rnd < num_rounds:
                    np.save(f"./fl_sessions/Session-{session}/round-{rnd}-weights.npy", aggregated_weights)
                elif rnd == num_rounds:
                    np.save(f"./fl_sessions/Session-{session}/global_session_model.npy", aggregated_weights)
            else:
                if rnd < num_rounds:
                    np.save(f"./fl_sessions/Session-{session}/round-{rnd}-weights.npy", aggregated_weights)
                elif rnd == num_rounds:
                    np.save(f"./fl_sessions/Session-{session}/global_session_model.npy", aggregated_weights)

        # loop through the sent results and update contribution ( pairs of key, value where
        # the key is the client id and the value is a dict of data size,sent size
        # and num_rounds_participated: updated value



# Define batch-size, nb of epochs and verbose for fitting
def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        with open('config_training.json', 'r') as config_training:
            config = config_training.read()
            data = json.loads(config)
            session = data['session']
        config = {
            "batch_size": 32,
            "epochs": 50,
            "verbose": 0,
            "rnd": rnd,
            "session": session
        }
        return config

    return fit_config


# Define hyper-parameters for evaluation
def evaluate_config(rnd: int):
    val_steps = 5 if rnd < 4 else 10
    with open('config_training.json', 'r') as config_training:
        config = config_training.read()
        data = json.loads(config)
        session = data['session']
    return {"val_steps": val_steps, "verbose": 0, "rnd": rnd, "session": session}
