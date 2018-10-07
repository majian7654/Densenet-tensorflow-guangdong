import tensorflow as tf
import input_data
from densenet import DenseNet

# Hyperparameter
growth_k = 24
nb_block = 2 # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-4
epsilon = 1e-4 # AdamOptimizer epsilon
dropoutRate = 0.2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
batch_size = 32
iteration = 63

#num_data = 2000
# batch_size * iteration = data_set_number

image_size_width = 256
image_size_height = 192

img_channels = 3
class_num = 12

total_epochs = 30
def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_ / 10.0
        test_acc += acc_ / 10.0

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary



def train():
    
    #prepare data
    image_batches, label_batches  = input_data.read_lvData('./data/label.csv')
    
    # image_size = 32, img_channels = 3, class_num = 10 in cifar10
    x = tf.placeholder(tf.float32, shape=[None, image_size_height, image_size_width, img_channels])
    label = tf.placeholder(tf.float32, shape=[None, class_num])
    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    logits = DenseNet(x=x, n_classes = class_num, nb_blocks=nb_block, filters=growth_k, dropout_rate = dropoutRate, training=training_flag).model
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    """
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
    train = optimizer.minimize(cost + l2_loss * weight_decay)

    In paper, use MomentumOptimizer
    init_learning_rate = 0.1

    but, I'll use AdamOptimizer
    """

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    train = optimizer.minimize(cost)


    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.image('input', x, 2)
    tf.summary.scale('acc:', accuracy)

    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./logs', sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        epoch_learning_rate = init_learning_rate
        for epoch in range(1, total_epochs + 1):
            if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
                epoch_learning_rate = epoch_learning_rate / 10

            for step in range(1, iteration + 1):
                image_batch,label_batch = sess.run([image_batches, label_batches])
                train_feed_dict = {
                    x: image_batch,
                    label: label_batch,
                    learning_rate: epoch_learning_rate,
                    training_flag : True
                }
                _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
                
                batch_acc = sess.run(accuracy,feed_dict=train_feed_dict )
                summary_str = sess.run(summary_op, feed_dict=train_feed_dict)
                
                summary_writer.add_summary(summary_str, step)

                #if step == iteration :
                #    train_loss /= iteration # average loss
                #    train_acc /= iteration # average accuracy
                #    train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                #                                      tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
                #    test_acc, test_loss, test_summary = Evaluate(sess)
                #    summary_writer.add_summary(summary=train_summary, global_step=epoch)
                #    summary_writer.add_summary(summary=test_summary, global_step=epoch)
                #    summary_writer.flush()
                #    line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                #        epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
                #    print(line)
                #    with open('logs.txt', 'a') as f :
                #        f.write(line)
            saver.save(sess=sess, save_path='./model/dense.ckpt')


if __name__=='__main__':
    train()
