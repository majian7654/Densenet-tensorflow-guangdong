import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope

class DenseNet():
    def __init__(self, x, n_classes, nb_blocks, filters, dropout_rate, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.dropout_rate = dropout_rate
        self.n_classes = n_classes
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = self._Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = self._Relu(x)
            x = self._conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = self._Drop_out(x, rate=self.dropout_rate, training=self.training)

            x = self._Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = self._Relu(x)
            x = self._conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = self._Drop_out(x, rate=self.dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = self._Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = self._Relu(x)
            x = self._conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = self._Drop_out(x, rate=self.dropout_rate, training=self.training)
            x = self._Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = self._Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = self._Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = self._conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
        # x = Max_Pooling(x, pool_size=[3,3], stride=2)


        """
        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))
        """




        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        x = self._Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = self._Relu(x)
        x = self._Global_Average_Pooling(x)
        x = flatten(x)
        x = self._Linear(x)


        # x = tf.reshape(x, [-1, 10])
        return x


    def _conv_layer(self,input, filter, kernel, stride=1, layer_name="conv"):
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network


    def _Global_Average_Pooling(self, x, stride=1):
        """
        width = np.shape(x)[1]
        height = np.shape(x)[2]
        pool_size = [width, height]
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
        It is global average pooling without tflearn
        """

        return global_avg_pool(x, name='Global_avg_pooling')
        # But maybe you need to install h5py and curses or not


    def _Batch_Normalization(self, x, training, scope):
        with arg_scope([batch_norm],
                       scope=scope,
                       updates_collections=None,
                       decay=0.9,
                       center=True,
                       scale=True,
                       zero_debias_moving_mean=True) :
            return tf.cond(training,
                           lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                           lambda : batch_norm(inputs=x, is_training=training, reuse=True))

    def _Drop_out(self, x, rate, training) :
        return tf.layers.dropout(inputs=x, rate=rate, training=training)

    def _Relu(self, x):
        return tf.nn.relu(x)

    def _Average_pooling(self, x, pool_size=[2,2], stride=2, padding='VALID'):
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


    def _Max_Pooling(self, x, pool_size=[3,3], stride=2, padding='VALID'):
        return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def _Concatenation(self,layers) :
        return tf.concat(layers, axis=3)

    def _Linear(self,x) :
        return tf.layers.dense(inputs=x, units=self.n_classes, name='linear')

