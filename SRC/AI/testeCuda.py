import tensorflow as tf

# Check if TensorFlow is built with CUDA
print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())

# Check for available devices, including GPUs
print("Available devices:", tf.config.list_physical_devices())

# Alternatively, check for GPU devices specifically
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs are available:", gpus)
else:
    print("No GPUs found")