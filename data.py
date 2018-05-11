import tensorflow as tf
import os
import numpy as np

# dset1_train_folder = '/data/DL_HW2/dset1/train'
# dset1_val_folder = '/data/DL_HW2/dset1/val'
  
def data_process(root_path):
    image_paths = []
    labels = []
    classes = sorted(os.walk(root_path).__next__()[1])
    for folder in classes:  
        folder_path = os.path.join(root_path, folder)
        walk = os.walk(folder_path).__next__()[2]
        for sample in walk:
            if sample.endswith('.jpg'):
                image_paths.append(os.path.join(folder_path, sample))
                category = int(folder[5:])
                labels.append(category)

    image_paths = tf.convert_to_tensor(image_paths, tf.string)  
    labels = tf.convert_to_tensor(labels, tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def _parse_function(image_path, label):
        image_string = tf.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [224, 224])
        image_normed = image_resized/255
        return image_normed, label
    
    dataset = dataset.map(_parse_function)
    return dataset
