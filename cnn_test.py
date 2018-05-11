import os
import cv2.cv2 as cv2
import numpy as np
import tensorflow as tf

# configs
FLAGS = tf.app.flags.FLAGS
# mode
tf.app.flags.DEFINE_boolean('is_training', True, 'training or testing')
# data
tf.app.flags.DEFINE_string('root_dir', '/data/DL_HW2', 'data root dir')
tf.app.flags.DEFINE_string('dataset', 'dset1', 'dset1 or dset2')
tf.app.flags.DEFINE_integer('n_label', 65, 'number of classes')
# trainig
tf.app.flags.DEFINE_integer('batch_size', 64, 'mini batch for a training iter')
tf.app.flags.DEFINE_string('save_dir', './checkpoints', 'dir to the trained model')
# test
tf.app.flags.DEFINE_string('my_best_model', './checkpoints/model.ckpt-1000', 'for test')

'''TODO: you may add more configs such as base learning rate, max_iteration,
display_iteration, valid_iteration and etc. '''

# hyperparameters
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_integer('max_iteration', 1000, 'number of batch for training')
tf.app.flags.DEFINE_integer('display_iteration', 100, 'display the loss and accuracy on train set')
tf.app.flags.DEFINE_integer('valid_iteration', 100, 'display the loss and accuracy on validation set')


class DataSet(object):
    '''
    Args:
        data_aug: False for valid/testing.
        shuffle: true for training, False for valid/test.
    '''
    def __init__(self, root_dir, dataset, sub_set, batch_size, n_label,
                 data_aug=False, shuffle=True):
        np.random.seed(0)
        self.data_dir = os.path.join(root_dir, dataset, sub_set)
        self.batch_size = batch_size
        self.n_label = n_label
        self.data_aug = data_aug
        self.shuffle = shuffle
        self.xs, self.ys = self.load_data()
        self._num_examples = len(self.xs)
        self.init_epoch()

    def load_data(self):
        '''Fetch all data into a list'''
        '''TODO: 1. You may make it more memory efficient if there is a OOM problem on
        you machine. 2. You may use data augmentation tricks.'''
        xs = []
        ys = []
        label_dirs = os.listdir(self.data_dir)
        label_dirs.sort()
        for _label_dir in label_dirs:
            print('loaded {}'.format(_label_dir))
            category = int(_label_dir[5:])
            label = np.zeros(self.n_label)
            label[category] = 1
            imgs_name = os.listdir(os.path.join(self.data_dir, _label_dir))
            imgs_name.sort()
            for img_name in imgs_name:
                im_ar = cv2.imread(os.path.join(self.data_dir, _label_dir, img_name))
                im_ar = cv2.cvtColor(im_ar, cv2.COLOR_BGR2RGB)
                im_ar = np.asarray(im_ar)
                im_ar = self.preprocess(im_ar)
                xs.append(im_ar)
                ys.append(label)
        return xs, ys

    def preprocess(self, im_ar):
        '''Resize raw image to a fixed size, and scale the pixel intensities.'''
        '''TODO: you may add data augmentation methods.'''
        im_ar = cv2.resize(im_ar, (224, 224))
        im_ar = im_ar / 255.0
        return im_ar

    def next_batch(self):
        '''Fetch the next batch of images and labels.'''
        if not self.has_next_batch():
            return None
        print(self.cur_index)
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            x_batch.append(self.xs[self.indices[self.cur_index+i]])
            y_batch.append(self.ys[self.indices[self.cur_index+i]])
        self.cur_index += self.batch_size
        return np.asarray(x_batch), np.asarray(y_batch)

    def has_next_batch(self):
        '''Call this function before fetching the next batch.
        If no batch left, a training epoch is over.'''
        start = self.cur_index
        end = self.batch_size + start
        if end > self._num_examples: 
            return False
        else: 
            return True

    def init_epoch(self):
        '''Make sure you would shuffle the training set before the next epoch.
        e.g. if not train_set.has_next_batch(): train_set.init_epoch()'''
        self.cur_index = 0
        self.indices = np.arange(self._num_examples)
        if self.shuffle:
            np.random.shuffle(self.indices)


class Model(object):
    def __init__(self):
        '''TODO: construct your model here.'''
        # Placeholders for input ims and labels
        self.ims = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
        self.labels = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.n_label])
        self.labels_shape = tf.shape(self.labels)
        self.keep_prob = tf.placeholder(tf.float32)

        # Construct model
        self.logits = self.construct_model()
        self.prediction = tf.nn.softmax(self.logits)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # init a tf session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def conv2d(self, x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, [1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        x = tf.nn.relu(x)
        return x

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

    def construct_model(self):
        '''TODO: Your code here.'''
        with tf.variable_scope('conv1'):
            self.conv1_kernel = tf.get_variable(name='kernels', 
                                           shape=[3, 3, 3, 32], 
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(stddev=0.05))          
            self.conv1_bias = tf.get_variable(name='bias', 
                                           shape=[32], 
                                           dtype=tf.float32,
                                           initializer=tf.constant_initializer(0.0))
            conv1 = self.conv2d(self.ims, self.conv1_kernel, self.conv1_bias)


        pool1 = self.maxpool2d(conv1)
        
        with tf.variable_scope('conv2'):
            self.conv2_kernel = tf.get_variable(name='kernels', 
                                           shape=[3, 3, 32, 64], 
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(stddev=0.05))
            self.conv2_bias = tf.get_variable(name='bias', 
                                           shape=[64], 
                                           dtype=tf.float32,
                                           initializer=tf.constant_initializer(0.0))
            conv2 = self.conv2d(pool1, self.conv2_kernel, self.conv2_bias)

        pool2 = self.maxpool2d(conv2)
        
        with tf.variable_scope('conv3'):
            self.conv3_kernel = tf.get_variable(name='kernels', 
                                           shape=[3, 3, 64, 128], 
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(stddev=0.05))
            self.conv3_bias = tf.get_variable(name='bias', 
                                           shape=[128], 
                                           dtype=tf.float32,
                                           initializer=tf.constant_initializer(0.0))
            conv3 = self.conv2d(pool2, self.conv3_kernel, self.conv3_bias)

        pool3 = self.maxpool2d(conv3)

        pool_reshape = pool3.get_shape().as_list()
        nodes = pool_reshape[1]*pool_reshape[2]*pool_reshape[3]
        reshaped_output = tf.reshape(pool3, [-1, nodes])
        # reshaped_output = tf.contrib.layers.flatten(pool3)

        with tf.variable_scope('full1'):
            self.full_weight1 = tf.get_variable(name='weights', 
                                           shape=[nodes, 1000], 
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(stddev=0.05))
            self.full_bias1 = tf.get_variable(name='bias', 
                                           shape=[1000], 
                                           dtype=tf.float32,
                                           initializer=tf.constant_initializer(0.0))
            full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, self.full_weight1), self.full_bias1))
            full_layer1 = tf.nn.dropout(full_layer1, self.keep_prob)

        with tf.variable_scope('full2'):
            self.full_weight2 = tf.get_variable(name='weights', 
                                           shape=[1000, 65], 
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(stddev=0.05))
            self.full_bias2 = tf.get_variable(name='bias', 
                                           shape=[65], 
                                           dtype=tf.float32,
                                           initializer=tf.constant_initializer(0.0))
            logits = tf.add(tf.matmul(full_layer1, self.full_weight2), self.full_bias2)

        return logits

    def train(self, ims, labels):
        '''TODO: Your code here.'''
        _, loss, acc = self.sess.run([self.train_op, self.loss_op, self.accuracy], feed_dict={self.ims: ims, self.labels: labels, self.keep_prob: 0.8})
        return loss, acc

    def valid(self, ims, labels):
        '''TODO: Your code here.'''
        loss, acc = self.sess.run([self.loss_op, self.accuracy], feed_dict={self.ims: ims, self.labels: labels, self.keep_prob: 1.0})
        return loss, acc

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir)

    def load(self):
        print('load model:', FLAGS.my_best_model)
        self.saver.restore(self.sess, FLAGS.my_best_model)


def train_wrapper(model):
    '''Data loader'''
    train_set = DataSet(FLAGS.root_dir, FLAGS.dataset, 'train',
                        FLAGS.batch_size, FLAGS.n_label,
                        data_aug=False, shuffle=True)
    valid_set = DataSet(FLAGS.root_dir, FLAGS.dataset, 'valid',
                        FLAGS.batch_size, FLAGS.n_label,
                        data_aug=False, shuffle=False)
    '''create a tf session for training and validation
    TODO: to run your model, you may call model.train(), model.save(), model.valid()'''
    best_accuracy = 0
    for step in range(1, FLAGS.max_iteration+1):
        if not train_set.has_next_batch():
            train_set.init_epoch()     
        batch_x, batch_y = train_set.next_batch()
        if len(batch_x) == FLAGS.batch_size:
            loss, acc = model.train(batch_x, batch_y)
            print("Step " + str(step) + ", Minibatch Loss= " + \
            "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
        if step % FLAGS.valid_iteration == 0:
            tot_acc = 0.0
            tot_input = 0
            while valid_set.has_next_batch():
                valid_ims, valid_labels = valid_set.next_batch()
                loss, acc = model.valid(valid_ims, valid_labels)
                tot_acc += acc*len(valid_ims)
                tot_input += len(valid_ims)
            acc = tot_acc / tot_input
            print("Current Accuracy= " + "{:.3f}".format(acc))            
            if acc > best_accuracy:
                model.save()
                best_accuracy = acc

    print("Optimization Finished!")    


def test_wrapper(model):
    '''Finish this function so that TA could test your code easily.'''    
    test_set = DataSet(FLAGS.root_dir, FLAGS.dataset, 'test',
                       FLAGS.batch_size, FLAGS.n_label,
                       data_aug=False, shuffle=False)
    '''TODO: Your code here.'''


def main(argv=None):
    print('Initializing models')
    model = Model()
    if FLAGS.is_training:
        train_wrapper(model)
    else:
        test_wrapper(model)


if __name__ == '__main__':
    tf.app.run()

