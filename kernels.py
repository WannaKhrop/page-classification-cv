from keras.models import load_model
import matplotlib.pyplot as plt

# load a model
model = load_model("model.keras")

# get the Conv2D layer and extract kernels
conv_layer = model.layers[0]
kernels = conv_layer.get_weights()[0]  # Extract the kernels

# Normalize the kernel values for better visualization
kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

n_filters, index = kernels.shape[-1], 1
for i in range(n_filters):
    f = kernels[:, :, :, i]
    ax = plt.subplot(n_filters // 8 + 1, 8, index)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(f[:, :, 0], cmap='gray')
    index += 1

plt.show()