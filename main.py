import src.utils as utils
from train_NN import gesture_NN as gesture_NN

# set dirs hierarchy in dict
dir = utils.get_dirs()

# train Gesture CNN
gesture_NN(dir, batch_size=128)
