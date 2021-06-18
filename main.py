import src.utils as utils
from train_NN import tr_gesture_NN as tr_gesture_NN
from test_NN import te_gesture_NN as te_gesture_NN

# set dirs hierarchy in dict
dir = utils.get_dirs()

# train Gesture CNN
tr_gesture_NN(dir, exp_env=True, batch_size=64)

# test Gesture CNN
# te_gesture_NN(dir, batch_size=64)
