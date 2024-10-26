DO_DATA_PARALLEL = True

PER_DEVICE_BATCH_SIZE = [2, 4, 8]
LEARNING_RATE = [1e-3, 1e-4]
MAX_N_EPOCHS = 10

TRAIN_DATA_SIZE = 1000
VALID_DATA_SIZE = 200
TEST_DATA_SIZE = 200

NUM_CLASSES = 10

DEVICE = "cuda"

MEMORY_LIMIT = 1.0

VISIBLE_DEVICES = [0, 1, 2, 3]

NUM_GPU = 1

NUM_SAMPLES = 2

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_ID = "2"
