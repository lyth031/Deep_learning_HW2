import tensorflow as tf
from data import data_process
from model import cnn_model
# Training Parameters
learning_rate = 0.001
epochs = 10
batch_size = 50
display_step = 10
num_steps = 1000
# Network Parameters
num_classes = 65 # total classes
# dropout = 0.5 # Dropout, probability to keep units

#Image path
dset1_train_folder = '/data/DL_HW2/dset1/train'
dset1_val_folder = '/data/DL_HW2/dset1/val'

train_data = data_process(dset1_train_folder)
val_data = data_process(dset1_val_folder)

train_data = train_data.shuffle(buffer_size=1000).batch(batch_size).repeat(epochs)
iterator = train_data.make_one_shot_iterator()

images = tf.placeholder(tf.float32, [None, 224, 224, 3]) # data input (img shape: 224*224*3)
labels = tf.placeholder(tf.float32, [None, num_classes])
# keep_prob = tf.placeholder(tf.float32)

# Construct model
logits = cnn_model(images)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for step in range(1, num_steps+1):       
        batch_x, batch_y = iterator.get_next()
        sess.run(train_op, feed_dict={images: batch_x, labels: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={images: batch_x, labels: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
            "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
    print("Optimization Finished!")
