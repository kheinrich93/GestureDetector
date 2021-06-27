import src.utils as utils
from train_NN import tr_gesture_NN
from predict import predict
from hp_tuner import find_opt_hp
from hp.hyperparams import hyperparams

# set dirs hierarchy in dict
dir = utils.get_dirs()

hp = hyperparams()

# train Gesture CNN
if hp.train_network:
    tr_gesture_NN(dir, hp, use_pretrained_cp=False, save_cp=True)

# find optimal hyperparams
if hp.tune_hp:
    find_opt_hp(dir, batch_size=64)

# test Gesture CNN
if hp.predict:
    predict(dir, hp)
