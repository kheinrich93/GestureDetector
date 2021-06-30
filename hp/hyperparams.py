class hyperparams:
    def __init__(self):
        self.epochs = 8
        self.batch_size = 64
        self.input_dim = (80, 80)
        self.n_classes = 7
        self.val_split = 0.2
        self.scale_factor = 255.0
        self.color_mode = 'rgb'  # grayscale or rgb

        self.train_network = False
        self.predict = True
        self.tune_hp = False
