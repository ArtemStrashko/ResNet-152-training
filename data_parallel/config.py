# Define hyperparameters
do_data_parallel = True

per_device_batch_size = 1
learning_rate = 2e-5
epochs = 5

train_data_size = 1000
test_data_size = 100
max_length = 512
model_name = 'resnet152'

device = 'cuda'

# Flag used to simulate limited memory. Set to 1.0 if you wish to use 100% memory on each device
memory_limit = 1.0

# Only use the specified devices
visible_devices = [0,1,2,3]