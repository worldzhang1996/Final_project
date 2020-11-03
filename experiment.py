# using the experiment to control parameters
class experiment:
    def __init__(self,distance_path = './data/trip_data_1_normalized_distance_trip.csv',time_path='./data/trip_data_1time_trip.csv'):
        self.lr = 0.01
        self.batch_size = 64
        self.momenta = 0.9
        self.weight_decay= 1e-5
        self.distance_path = distance_path
        self.time_path = time_path

        self.epochs = 10
