# first implementation of a training loop
import os
import sys
from typing import List
from pydantic import BaseModel
import torch
import numpy as np
sys.path.append(os.getcwd())

from msc.encoder import Encoder
from stable_baselines3 import PPO
from decoder import Decoder
import gym
from buffer import Decoder_Buffer


class Trainer(BaseModel):
    """A Trainer class for the combined model
    """
    # general parameters
    latent_dim: int = 128
    image_dim: List[int] = None

    # rl agent parameters
    environment_name: str =  "BreakoutNoFrameskip-v4"
    encoder_class = Encoder
    rl_agent_type = PPO
    rl_agent_train_steps: int = 10

    # Decoder Parameters
    decoder_train_steps: int = 10 # TODO: look over class variables
    decoder_class: object = None # TODO: add class

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
        self._rl_agent = self.rl_agent_type("CnnPolicy", self.environment_name, policy_kwargs=policy_kwargs, verbose=1)
        self._env = self._rl_agent.env # for easy access to env
        self.image_dim = self._env.reset().shape[1:] 
        self._encoder_network = self._rl_agent.policy.features_extractor # for easy access to encoder

    def init_Decoder(self):
        dimension = self._buffer.get_image_dims()
        self._Decoder = Decoder(self.latent_dim, dimension[0]) # TODO: check how to use feature dims

    def init_buffer(self):
        self._buffer = self.buffer_class(list(self.image_dim), self.latent_dim, self.buffer_size, to_gray=True, flatten=True)


#---------------------- Train methods -----------------------------------------        

    def train_rl_agent(self):
        self._rl_agent.learn(self.rl_agent_train_steps)
        

    def train_Decoder(self, epochs, batch_size):
        # TODO add images
        self._Decoder.train(self._buffer, epochs, batch_size)

    def train(self):
        # self.train_rl_agent() # train latent network (encoder)
        self.fill_buffer_randomly(100) # fill buffer
        self.train_Decoder(50, 10) # train decoder

    def fill_buffer_randomly(self, steps=100):
        """Fills the buffer randomly

        Args:
            steps (int): Number of steps to take
        """
        observation = self._env.reset()
        for i in range(steps):
            action = self._rl_agent.predict(observation)#self._env.action_space.sample() # your agent here (this takes random actions)
            observation, reward, done, info = self._env.step(action)
            observation = torch.from_numpy(observation).float()
            latent = self._encoder_network(observation).detach() # make sure to detach the latents to not propagate back through rl agent in decoder training
            self._buffer.add(observation, latent)
            if done:
                observation = self._env.reset()

        self._env.close()

    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
            



t = Trainer() # here you can put args in trainer
t.train()
# t.fill_buffer_randomly()



# next steps:
# - improve decoder
# - try first training loop
# - give encoder to decoder


# tasks:

# till tuesday
# - get running example (flattened image) (Philipp)
# - write his report
# - draw architecture
# - text him with david and ask for gpu for training

# rest until meeting
# - method to save/load RL agent (and Decoder)
# - look in convolution AE
# - look back to training of AE?
# - implement second trainig method
# - look for a good environment + agent
# - try other metric
