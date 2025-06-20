import tensorflow as tf
import tensorflow_datasets as tfds

# Load the MNIST dataset
(pcap_data_train, pcap_data_test), pcap_ds_info = tfds.load(
    'mnist', split=['train', 'test'],
    shuffle_files=True, as_supervised=True,
    with_info=True
)

def pcap_normalize_data(image, label):
    return tf.cast(image, tf.float32) / 255., label

# Apply normalization to both training and test datasets
pcap_data_train = pcap_data_train.map(
    pcap_normalize_data,
    num_parallel_calls=tf.data.AUTOTUNE
)
pcap_data_test = pcap_data_test.map(
    pcap_normalize_data,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Train the model
model.fit(
    pcap_data_train.batch(32),
    epochs=6,
    validation_data=pcap_data_test.batch(32)
)

# Make predictions
predictions = model.predict(pcap_data_test.batch(32))
print("Predictions shape:", predictions.shape)
