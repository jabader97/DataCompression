from pytorch_msssim import ms_ssim
import numpy as np


data = np.load('DataCompression/dat/digits389.npy', allow_pickle=True).item()
x_test = data['Xtest']
x_train = data['Xtrain']
y_test = data['Ytest']
y_train = data['Ytrain']