import sys, os
from tqdm import tqdm
sys.path.append("C:/Users/Philipp von Bachmann/Documents/Dokumente_Philipp/Uni/DataCompression")
sys.path.append(os.getcwd())
from DataCompression.src.training_loop import Trainer
from DataCompression.src.metric import evaluate
config = {
"rl_agent_train_steps": 10000,
"buffer_size": 1000,
"added_error": 1e-3,
"alpha": 1e-4,
}
mytrainer = Trainer(**config)
mytrainer.train_rl_agent(10000, save_every=None)
mytrainer.fill_buffer(1000)

evaluate(mytrainer._encoder_network, None,mytrainer._buffer, True, n_digits=3)

