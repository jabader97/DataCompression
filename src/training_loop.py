# first implementation of a training loop
import os
import sys
from typing import List
from pydantic import BaseModel
import torch
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
    decoder_class = None # TODO: add class

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
        self.init_Decoder()
        self.init_buffer()

        
    def init_rl_agent(self):
        """Init RL agent
        """
        policy_kwargs = dict(
        features_extractor_class=self.encoder_class,
        features_extractor_kwargs=dict(features_dim=self.latent_dim)) # defines output feature size
        self._rl_agent = self.rl_agent_type("CnnPolicy", self.environment_name, policy_kwargs=policy_kwargs, verbose=1)
        self._env = self._rl_agent.env # for easy access to env
        self.image_dim = self._env.reset().shape[1:] # TODO: check # 3 dim, actual tensor will be n_sample cross those 3
        self._encoder_network = self._rl_agent.policy.features_extractor # for easy access to encoder

    def init_Decoder(self):
        self._Decoder = Decoder(self.latent_dim, self.image_dim[-1]) # TODO: check how to use feature dims

    def init_buffer(self):
        self._buffer = self.buffer_class(list(self.image_dim), self.latent_dim, self.buffer_size)

#---------------------- Train methods -----------------------------------------        

    def train_rl_agent(self):
        self._rl_agent.learn(self.rl_agent_train_steps)
        

    def train_Decoder(self, epochs, batch_size):
        # TODO add images
        self._Decoder.train(self._buffer, epochs, batch_size)

    def train(self):
        self.train_rl_agent() # train latent network (encoder)
        self.fill_buffer_randomly(10000) # fill buffer
        self.train_Decoder() # train decoder

    def fill_buffer_randomly(self, steps):
        """Fills the buffer randomly

        Args:
            steps (int): Number of steps to take
        """
        observation = self._env.reset()
        for i in range(steps):
            action = self._rl_agent.predict(observation)#self._env.action_space.sample() # your agent here (this takes random actions)
            observation, reward, done, info = self._env.step(action)
            observation = torch.from_numpy(observation).float()
            # TODO: flatten observation, this also changes the buffer
            latent = self._encoder_network(observation)
            self._buffer.add(observation, latent)

            if done:
                observation = self._env.reset()

        self._env.close()
            



t = Trainer() # here you can put args in trainer
t.train()


# next steps:
# - improve decoder
# - try first training loop
# - give encoder to decoder


# tasks:

# till tuesday
# - get running example (flattened image) (Philipp)
# - write his report
# - draw architecture

# rest until meeting
# - look in convolution AE
# - look back to training of AE?
# - implement second trainig method
# - look for a good environment + agent
# - try other metric
