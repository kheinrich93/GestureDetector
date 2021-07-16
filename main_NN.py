from train_NN import tr_gesture_NN
from test_NN import te_gesture_NN
import src.utils as utils
from hp_tuner import find_opt_hp
from hp.hyperparams import hyperparams

# TODO
# visualize input with labels and results
# testing
# predict on own imgs and pre-processed asl

# set dirs hierarchy in dict
dirs = utils.get_dirs()

# load hyperparams
hp = hyperparams()

# train Gesture CNN
if hp.train_network:
    tr_gesture_NN(dirs, hp, save_weights_as='mnist2', use_pretrained_cp=False,
                  save_cp=True, dataset='mnist')

if hp.test_network:
    te_gesture_NN(dirs, hp, 'mnist', 'cp_gesture_mnist')

# find optimal hyperparams
if hp.tune_hp:
    find_opt_hp(dirs, batch_size=64)
