# first implementation of a training loop
import os, time
import sys
import json
from typing import List, Any
from pydantic import BaseModel
import torch
import numpy as np
sys.path.append(os.getcwd())
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from msc.encoder import Encoder
from stable_baselines3 import PPO
import gym
from tqdm import tqdm
from src.buffer import Decoder_Buffer
from matplotlib import pyplot as plt
import src.decoder_MSE as Decoder

class Trainer(BaseModel):
    """A Trainer class for the combined model
    """
    # general parameters
    latent_dim: int = 128
    image_dim: List[int] = None
    save_path: str = "P:/Dokumente/3 Uni/SoSe21/Data_Compression/DataCompression" # my save path ;) 

    # rl agent parameters
    environment_name: str =  "BreakoutNoFrameskip-v4"
    encoder_class: Any = Encoder # at the moment, you cannot change this variable!!
    use_custom_encoder: bool = True
    rl_agent_type: object = PPO
    rl_agent_train_steps: int = 10

    # Decoder Parameters
    decoder_train_steps: int = 10 # TODO: look over class variables
    decoder_batch_size: int = 100
    decoder_class: object = None # TODO: add class
    decoder_model: object = None
    decoder_loss: object = torch.nn.MSELoss()
    decoder_lr: float = 1e-4
    decoder_optimizer: object = None

    # buffer
    buffer_class = Decoder_Buffer
    buffer_size: int = 1000 # dont write 1e4, need int not float

    # internal variables for tracking
    _encoder_network: object = None
    _env: object = None
    _rl_agent: object = None
    _Decoder: object = None
    _buffer: object = None


    # pydantic config
    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True
        underscore_attrs_are_private = True

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.init_rl_agent() # init rl agent
        self.init_buffer()
        self.init_Decoder()

        
    def init_rl_agent(self):
        """Init RL agent
        """
        policy_kwargs = dict(
        features_extractor_class=self.encoder_class,
        features_extractor_kwargs=dict(features_dim=self.latent_dim)) # defines output feature size
        if not self.use_custom_encoder:
            from stable_baselines3.common.torch_layers import NatureCNN
            policy_kwargs["features_extractor_class"] = NatureCNN
            print("Using default encoder from environment")

        self._rl_agent = self.rl_agent_type("CnnPolicy", self.environment_name, policy_kwargs=policy_kwargs, verbose=1)
        self._env = self._rl_agent.env # for easy access to env
        self.image_dim = self._env.reset().shape[1:] 
        self._encoder_network = self._rl_agent.policy.features_extractor # for easy access to encoder

    def init_Decoder(self):
        if self.decoder_model is None:
            dimension = self._buffer.get_image_dims()
            self.decoder_model = Decoder.AE(self.latent_dim, dimension[0])
        if self.decoder_optimizer is None:
            self.decoder_optimizer = torch.optim.Adam(self.decoder_model.parameters(), lr=self.decoder_lr)
        self._Decoder = Decoder.Decoder(self.decoder_model, self.decoder_loss, self.decoder_optimizer)  # TODO: check how to use feature dims

    def init_buffer(self):
        self._buffer = self.buffer_class(list(self.image_dim), self.latent_dim, self.buffer_size, to_gray=False, flatten=False)


#---------------------- Train methods -----------------------------------------        

    def train_rl_agent(self, epochs, save_every=None):
        if save_every:
            for i in range(0, epochs, save_every):
                self._rl_agent.learn(save_every)
                self._rl_agent.save(self.save_path + "rl_agent")
                print("\n Saved RL agent")
        else:
            self._rl_agent.learn(epochs)
        

    def train_Decoder(self, epochs, batch_size, save_every=None):
        # TODO add images
        if save_every:
            epoch_losses=[]
            for i in range(0, epochs, save_every):
                epoch_losses += self._Decoder.train(self._buffer, save_every, batch_size)
                self._Decoder.save(self.save_path)
        else:
            epoch_losses = self._Decoder.train(self._buffer, epochs, batch_size)
        return epoch_losses

    def train(self):
        self.train_rl_agent(self.rl_agent_train_steps) # train latent network (encoder)
        self.fill_buffer(self.buffer_size) # fill buffer
        self.train_Decoder(self.decoder_train_steps, batch_size=self.decoder_batch_size) # train decoder

    def fill_buffer(self, samples=100, randomly=True, use_tqdm=True):
        """Fills the buffer randomly

        Args:
            steps (int): Number of steps to take
            use_tqdm (bool): Add progress bar if True.
        """
        print(f"Filling buffer with {samples} samples")
        time.sleep(1) # to avoid printstream clashing with progressbar
        observation = self._env.reset()
        r = tqdm(range(samples)) if use_tqdm else range(samples)
        get_action = self.random_env_action if randomly else self._rl_agent.predict
        for i in r:
            action = get_action(observation)
            observation, reward, done, info = self._env.step(action)
            observation = torch.from_numpy(observation).float()
            latent = self._encoder_network(observation).detach() # make sure to detach the latents to not propagate back through rl agent in decoder training
            self._buffer.add(observation, latent)
            if done:
                observation = self._env.reset()

        self._env.close()


    def random_env_action(self, dummy=None):
        """Chooses a random env action, mainly just wraps the action to the format expected by the agent
            Has dummy value to use same signature as Rl agent sample
        Returns:
            tuple(numpyp.array, None): The action
        """
        return (np.array([self._env.action_space.sample()], dtype=np.int64), None)

    def save(self, filepath=None, buffer=False):
        """Saves the rl_agent and the Decoder into the given directory

        Args:
            filepath (string): path to the directory
            buffer (bool): If the buffer should also be saved. Defaults to False.
        """
        if not filepath:
            filepath = self.save_path
        if not os.path.exists(filepath):
            os.mkdir(filepath)

        if filepath[-1] != "/":
            filepath += "/"

        self._rl_agent.save(filepath + "rl_agent") # save rl agent
        self._Decoder.save(filepath)
        if buffer:
            self._buffer.save(filepath)
        with open(filepath + "config.txt", "w") as f: # this just writes all variables whose types are known to pydantic
            json.dump(self.dict(), f, indent=4)

    def load(self, filepath=None, buffer=False):
        """Loads the rl_agent and the Decoder from the given directory.

        Args:
            filepath (string): path to the directory
        """
        if not filepath:
            filepath = self.save_path
        if filepath[-1] != "/":
            filepath += "/"
        self._rl_agent.load(filepath + "/rl_agent")
        self._Decoder.load(filepath)
        if buffer:
            self._buffer.load(filepath)





# tasks:

# till tuesday
# - get running example (flattened image) (Philipp) done
# - write his report
# - draw architecture
# - text him with david and ask for gpu for training done

# rest until meeting
# - method to save/load RL agent (and Decoder) done
# - look in convolution AE
# - look back to training of AE?
# - implement second trainig method
# - look for a good environment + agent
# - try other metric



# change first dimension when using grayscale (from 3 to 1 instead of None)
