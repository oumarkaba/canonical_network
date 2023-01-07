#%%
from canonical_network.prepare.rotated_mnist_data import *


ds = get_dataset('../data/rotated_mnist', setify=True)


ds[0]