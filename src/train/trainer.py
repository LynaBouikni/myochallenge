
import json
import os
from abc import ABC
from dataclasses import dataclass, field
from typing import List

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize




#A custom implementation of the Trainer class for training reinforcement learning agents using the Stable Baselines3 library.
#The MyoTrainer class inherits from Trainer and provides additional functionality specific to the Myo armband, such as saving/loading
#environment configurations and models, and specifying a custom policy type for the agent. The __main__ block at the end of the module
#provides a simple message indicating that the module should not be run directly, but rather should be imported and used in another Python script,  main.py.



# an abstract base class Trainer for training a library-independent RL algorithm on a gym environment
# requires the implementation of the _init_agent and train methods in any concrete subclass that inherits from this base class.  

class Trainer(ABC):
    """
    Protocol to train a library-independent RL algorithm on a gym environment.
    """

    envs: VecNormalize
    env_config: dict
    model_config: dict
    model_path: str
    total_timesteps: int
    log: bool

    def _init_agent(self) -> None:
        """Initialize the agent."""

    def train(self, total_timesteps: int) -> None:
        """Train agent on environment for total_timesteps episdodes."""


#main class "MyoTrainer": implements the Trainer protocol as an RL algorithm based on the Stable Baselines 3 library, 
#specifically the RecurrentPPO algorithm from the sb3_contrib package
#This module can be imported and used in other Python scripts to train RL agents on gym environments using Stable Baselines 3.

@dataclass
class MyoTrainer:

    #. The trainer must have the following attributes:

    envs: VecNormalize                                              #envs: A VecNormalize object containing the gym environment(s) to train the agent on.
    env_config: dict                                                #env_config: A dictionary of configuration parameters for the environment(s).
    load_model_path: str                                            #load_model_path: The path to a saved RL model to continue training from. If None, a new model will be initialized.                                        
    log_dir: str                                                    #log_dir: The directory to save training logs and the final trained model.
    model_config: dict = field(default_factory=dict)                #model_config: A dictionary of configuration parameters for the RL model.
    callbacks: List[BaseCallback] = field(default_factory=list)     #callbacks: A list of callback objects to use during training.         
    timesteps: int = 10_000_000                                     #timesteps: The total number of training timesteps to run.


    #constructor, initializes the RL agent and saves the environment configuration.
    def __post_init__(self):
        self.dump_env_config(path=self.log_dir)                
        self.agent = self._init_agent()

    #A method that saves the environment configuration to a JSON file.
    def dump_env_config(self, path: str) -> None:
        with open(os.path.join(path, "env_config.json"), "w", encoding="utf8") as f:
            json.dump(self.env_config, f)

    #A method that initializes the RL agent using the saved model or a new one.
    def _init_agent(self) -> RecurrentPPO:
        if self.load_model_path is not None:
            return RecurrentPPO.load(
                self.load_model_path,
                env=self.envs,
                tensorboard_log=self.log_dir,
                custom_objects=self.model_config,
            )
        print("\nNo model path provided. Initializing new model.\n")
        return RecurrentPPO(
            "MlpLstmPolicy",
            self.envs,
            verbose=2,
            tensorboard_log=self.log_dir,
            **self.model_config,
        )

    #A method that trains the RL agent for a specified number of timesteps, using the specified callbacks
    def train(self, total_timesteps: int) -> None:
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=self.callbacks,
            reset_num_timesteps=True,
        )

    #A method that saves the final trained model and environment to files.
    def save(self) -> None:
        self.agent.save(os.path.join(self.log_dir, "final_model.pkl"))
        self.envs.save(os.path.join(self.log_dir, "final_env.pkl"))


if __name__ == "__main__":
    print("This is a module. Run main.py to train the agent.")

