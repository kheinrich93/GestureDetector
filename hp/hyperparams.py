class hyperparams:
    def __init__(self):
        self.epochs = 8
        self.batch_size = 64
        self.input_dim = (28, 28)
        self.n_classes = 7
        self.val_split = 0.2
        self.scale_factor = 255.0
        self.color_mode = 'grayscale'  # grayscale or rgb

        self.train_network = True
        self.predict = False
        self.tune_hp = False

        # considered letters for mnist
        self.letters = ['A', 'B', 'C', 'H', 'K', 'L', 'O']
