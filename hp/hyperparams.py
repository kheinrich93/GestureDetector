class hyperparams:
    def __init__(self):
        self.epochs = 4
        self.batch_size = 64
        self.input_dim = (64, 64)
        self.n_classes = 29
        self.val_split = 0.2
        self.scale_factor = 255.0
        self.color_mode = 'grayscale'  # grayscale or rgb

        self.train_network = False
        self.predict = True
        self.tune_hp = False
