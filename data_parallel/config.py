DO_DATA_PARALLEL = True

PER_DEVICE_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_N_EPOCHS = 20

TRAIN_DATA_SIZE = 1000
VALID_DATA_SIZE = 200
TEST_DATA_SIZE = 200

DEVICE = "cuda"

# Flag used to simulate limited memory. Set to 1.0 if you wish to use 100% memory on each device
MEMORY_LIMIT = 1.0

# Only use the specified devices
VISIBLE_DEVICES = [0, 1, 2, 3]

MLFLOW_TRACKING_URI = "http://localhost:5001"
# MLFLOW_EXPERIMENT_ID = "1"
MLFLOW_EXPERIMENT_NAME = "resnet152"
MLFLOW_RUN_NAME = "data_parallel"
MLFLOW_PARENT_RUN = None
