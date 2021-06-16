import numpy as np
from DataCompression.msc.vae_agarap import AE
from DataCompression.src.decoder_mse import decode

# load the data
data = np.load('digits389.npy', allow_pickle=True).item()
x_test = data['Xtest']
x_train = data['Xtrain']
y_test = data['Ytest']
y_train = data['Ytrain']

# train the original vae to get the low dim spaces (this is slow because used single batches)
print("Original VAE loss:")
ae = AE()
ae.train_ae(x_train, epochs=2)
# get the lower dimensional space it produced
latent = ae.get_latent(x_train)

# reformat the data to fit the decoder method
latent = np.asarray(latent)

# train the decoder network
print("Our decoder network loss:")
decode(x_train, latent, epochs=20)

# after 2 epochs for compression network and 20 epochs for decompression network
# achieves loss of 0.128
