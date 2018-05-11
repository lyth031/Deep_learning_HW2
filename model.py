import tensorflow as tf

def cnn_model(input_images,train_logical=True):    
    with tf.variable_scope('conv1'):
        conv1_kernel = tf.get_variable(name='kernels', 
                                    shape=[3, 3, 3, 32], 
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.05))
        conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1, 1, 1, 1], padding='SAME')
        conv1_bias = tf.get_variable(name='bias', 
                                    shape=[32], 
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
        conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
        relu_conv1 = tf.nn.relu(conv1_add_bias)

    pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool_layer1')
    
    with tf.variable_scope('conv2'):
        conv2_kernel = tf.get_variable(name='kernels', 
                                    shape=[3, 3, 32, 64], 
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.05))
        conv2 = tf.nn.conv2d(pool1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
        conv2_bias = tf.get_variable(name='bias', 
                                    shape=[64], 
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
        conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
        relu_conv2 = tf.nn.relu(conv2_add_bias)

    pool2 = tf.nn.max_pool(relu_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool_layer2')
    
    with tf.variable_scope('conv3'):
        conv3_kernel = tf.get_variable(name='kernels', 
                                    shape=[3, 3, 64, 128], 
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.05))
        conv3 = tf.nn.conv2d(pool2, conv3_kernel, [1, 1, 1, 1], padding='SAME')
        conv3_bias = tf.get_variable(name='bias', 
                                    shape=[128], 
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
        conv3_add_bias = tf.nn.bias_add(conv3, conv3_bias)
        relu_conv3 = tf.nn.relu(conv3_add_bias)

    pool3 = tf.nn.max_pool(relu_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool_layer3')

    pool_reshape = pool3.get_shape().as_list()
    nodes = pool_reshape[1]*pool_reshape[2]*pool_reshape[3]
    reshaped_output = tf.reshape(pool3, [-1, nodes])

    with tf.variable_scope('full1'):
        full_weight1 = tf.get_variable(name='weights', 
                                    shape=[nodes, 65], 
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.05))
        full_bias1 = tf.get_variable(name='bias', 
                                    shape=[65], 
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weight1), full_bias1))

    logits = full_layer1

    return logits
