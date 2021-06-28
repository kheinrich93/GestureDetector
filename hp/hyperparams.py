class hyperparams:
    def __init__(self):
        self.epochs = 6
        self.batch_size = 64
        self.input_dim = (64, 64)
        self.n_classes = 29
        self.val_split = 0.2
        self.scale_factor = 255.0
        self.color_mode = 'rgb'  # grayscale or rgb

        self.train_network = True
        self.predict = False
        self.tune_hp = False
