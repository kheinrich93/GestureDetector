from train_NN import tr_gesture_NN
import src.utils as utils
from src.predict import predict_from_dir
from hp_tuner import find_opt_hp
from hp.hyperparams import hyperparams

# set dirs hierarchy in dict
dir = utils.get_dirs()

# load hyperparams
hp = hyperparams()

# train Gesture CNN
if hp.train_network:
    tr_gesture_NN(dir, hp, use_pretrained_cp=False,
                  save_cp=False, dataset='asl')

# find optimal hyperparams
if hp.tune_hp:
    find_opt_hp(dir, batch_size=64)

# test Gesture CNN
if hp.predict:
    predict_from_dir(dir, hp)
