import numpy as np
import binascii
import os
from array import array
from random import shuffle
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

# Configuration
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

# Function to convert pcap file to matrix
def getMatrixfrom_pcap(filename, width):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)
    fh = np.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])
    rn = len(fh)//width
    fh = np.reshape(fh[:rn*width], (-1, width))
    fh = np.uint8(fh)
    return fh

def prepare_dataset(names):
    for name in names:
        data_image = array('B')
        data_label = array('B')
        FileList = []
        
        # Get all PNG files
        for dirname in os.listdir(name[0]):
            path = os.path.join(name[0], dirname)
            for filename in os.listdir(path):
                if filename.endswith(".png"):
                    FileList.append(os.path.join(name[0], dirname, filename))
        
        shuffle(FileList)
        
        # Process each image
        for filename in FileList:
            label = int(filename.split(os.sep)[2])
            with Image.open(filename) as im:
                pixel = im.load()
                width, height = im.size
                
                # Check image size
                if max(width, height) > 256:
                    raise ValueError('Image exceeds maximum size: 256x256 pixels')
                
                # Append pixel data
                for x in range(width):
                    for y in range(height):
                        data_image.append(pixel[y, x])
                
                # Append label
                data_label.append(label)
        
        # Create header for labels
        hexval = f"{len(FileList):#0{6}x}".zfill(8)
        header = array('B')
        header.extend([0, 0, 8, 1])
        header.extend([int(hexval[i:i+2], 16) for i in range(2, 10, 2)])
        
        # Create header for images
        header.extend([0, 0, 0, width, 0, 0, 0, height])
        header[3] = 3  # Change MSB for image data
        
        # Combine data with headers
        data_image = header + data_image
        data_label = header + data_label
        
        # Save files
        with open(name[1] + '-images-idx3-ubyte', 'wb') as f:
            data_image.tofile(f)
        with open(name[1] + '-labels-idx1-ubyte', 'wb') as f:
            data_label.tofile(f)

def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

def main():
    # Example usage
    names = [
        ("path/to/train", "train"),
        ("path/to/test", "test")
    ]
    
    # Prepare dataset
    prepare_dataset(names)
    
    # Load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    # Train model
    model.fit(
        ds_train.batch(32),
        epochs=6,
        validation_data=ds_test.batch(32)
    )

if __name__ == "__main__":
    main()
