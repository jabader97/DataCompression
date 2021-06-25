import sys, os
sys.path.append(os.getcwd())
from src.training_loop import Trainer
filepath = f"P:/Dokumente/3 Uni/SoSe21/Data_Compression/DataCompression/exp/test_dataset/"
config = {"buffer_size": 1000, "use_custom_encoder": False}
mytrainer = Trainer(**config)
mytrainer.train_rl_agent(50000)
mytrainer._rl_agent.save(filepath + "rl_agent")
mytrainer.fill_buffer(1000, randomly=False)
mytrainer._buffer.save(filepath)