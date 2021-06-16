# issue: to use this package, image would need to have dimensions > 160 (this dataset is 16x16)

from pytorch_msssim import ms_ssim
import numpy as np
from matplotlib import pyplot as plt
import torch

# load digits
data = np.load('digits389.npy', allow_pickle=True).item()
x_test = data['Xtest']
x_train = data['Xtrain']
y_test = data['Ytest']
y_train = data['Ytrain']


# show one image
first_image = x_test[0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((16, 16))
plt.imshow(pixels, cmap='gray')
plt.show()
first_reformed = np.expand_dims(np.expand_dims(pixels, axis=0), axis=0)
first_reformed = torch.from_numpy(first_reformed)

# show a second image
second_image = x_test[500]
second_image = np.array(second_image, dtype='float')
pixels2 = second_image.reshape((16, 16))
plt.imshow(pixels2, cmap='gray')
plt.show()

# apply ms_ssim
print("Between image and itself")
print(1 - ms_ssim(first_reformed, first_reformed))
print("Between two images")
print(1 - ms_ssim(first_image, second_image))
print("Between two images (reversed, should be same as above)")
print(1 - ms_ssim(second_image, first_image))

