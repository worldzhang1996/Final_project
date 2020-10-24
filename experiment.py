# using the experiment to control parameters
class experiment:
    def __init__(self):
        self.lr = 0.1
        self.batch_size = 64
        self.momenta = 0.9
        self.weight_decay= 1e-5
        self.distance_path = "./data/distance_trip_data_1.csv"
        self.time_path = "./data/time_trip_data_1.csv"

        self.epochs = 10
