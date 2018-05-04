import tensorflow as tf
import os

dset1_train_folder = '/data/DL_HW2/dset1/train'
dset1_val_folder = '/data/DL_HW2/dset1/val'
  
def data_process(root_path):
    image_paths = []  
    labels = []  
    label = 0  
    classes = sorted(os.walk(root_path).__next__()[1])
    for folder in classes:  
        folder_path = os.path.join(root_path, folder)
        walk = os.walk(folder_path).__next__()[2]
        for sample in walk:
            if sample.endswith('.jpg'):
                image_paths.append(os.path.join(folder_path, sample))
                labels.append(label)
        label += 1

    image_paths = tf.convert_to_tensor(image_paths, tf.string)  
    labels = tf.convert_to_tensor(labels, tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def _parse_function(image_path, label):
        image_string = tf.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [224, 224])
        return image_resized, label
    
    dataset = dataset.map(_parse_function)
 
    dataset = dataset.shuffle(buffer_size=1000).batch(50).repeat(10)
    return dataset
train_data = data_process(dset1_train_folder)
test_data = data_process(dset1_val_folder)