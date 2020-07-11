from GAHNE.layers import *
from GAHNE.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()  # name = 'dhnedhne'
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.embeddings = None

        self.loss = 0
        self.accuracy = 0
        self.pred = 0
        self.actual = 0

        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """

        with tf.variable_scope(self.name):  # name = 'dhne'
            self._build()
        hidden1_1 = self.layers[0](self.inputs)
        hidden1_2 = self.layers[1](hidden1_1)
        hidden2_1 = self.layers[2](self.inputs)
        hidden2_2 = self.layers[3](hidden2_1)
        self.embeddings = tf.concat([hidden1_2, hidden2_2], 1) #hidden1_2
        self.outputs = self.layers[4](self.embeddings)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        self._micromacro()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def _micromacro(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GAHNE(Model):
    def __init__(self, placeholders, input_dim, total_num, **kwargs):
        super(GAHNE, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.total_num = total_num
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layers[1].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layers[2].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layers[3].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layers[4].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])


    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _micromacro(self):
        self.pred, self.actual = masked_pred(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])



    def _build(self):
        #################

        # self.layers.append(GraphTraditionalConvolution(input_dim=self.input_dim,
        #                                                output_dim=FLAGS.hidden1,
        #                                                placeholders=self.placeholders,
        #                                                act=tf.nn.relu,
        #                                                dropout=True,
        #                                                sparse_inputs=True,
        #                                                logging=self.logging))
        #
        # self.layers.append(GraphTraditionalConvolution(input_dim=FLAGS.hidden1,
        #                                                output_dim=FLAGS.hidden1,
        #                                                placeholders=self.placeholders,
        #                                                act=lambda x: x,
        #                                                dropout=True,
        #                                                logging=self.logging))

        #################
        #
        # self.layers.append(GraphQWTAttention(input_dim=self.input_dim,
        #                                     # hidden=FLAGS.hidden1,
        #                                     output_dim=FLAGS.hidden1,
        #                                     placeholders=self.placeholders,
        #                                     act=tf.nn.relu,
        #                                     bias=True,
        #                                     dropout=True,
        #                                     sparse_inputs=True,
        #                                     logging=self.logging))
        #
        # self.layers.append(GraphQWTAttention(input_dim=FLAGS.hidden1,
        #                                      output_dim=FLAGS.hidden1,
        #                                      placeholders=self.placeholders,
        #                                      act=lambda x: x,
        #                                      bias = True,
        #                                      dropout=True,
        #                                      logging=self.logging))
        #################
        self.layers.append(GraphWeight(input_dim=self.input_dim,
                                             output_dim=FLAGS.hidden1,
                                             placeholders=self.placeholders,
                                             act=tf.nn.relu,
                                             dropout=True,
                                             bias=True,
                                             sparse_inputs=True,
                                             logging=self.logging))

        self.layers.append(GraphWeight(input_dim=FLAGS.hidden1,
                                             output_dim=FLAGS.hidden1,
                                             placeholders=self.placeholders,
                                             act=lambda x: x,
                                             dropout=True,
                                             bias=True,
                                             logging=self.logging))

        #################
        # self.layers.append(GraphGate(input_dim=self.input_dim,
        #                                      output_dim=FLAGS.hidden1,
        #                                      placeholders=self.placeholders,
        #                                      act=tf.nn.relu,
        #                                      dropout=True,
        #                                      bias=True,
        #                                      sparse_inputs=True,
        #                                      logging=self.logging))
        #
        # self.layers.append(GraphGate(input_dim=FLAGS.hidden1,
        #                                      output_dim=FLAGS.hidden1,
        #                                      placeholders=self.placeholders,
        #                                      act=lambda x: x,
        #                                      dropout=True,
        #                                      bias=True,
        #                                      logging=self.logging))

        #################
        self.layers.append(GraphTraditionalConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphTraditionalConvolution(input_dim=FLAGS.hidden1,
                                                       output_dim=FLAGS.hidden1,
                                                       placeholders=self.placeholders,
                                                       act=lambda x: x,
                                                       dropout=True,
                                                       logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1 * 2,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 bias=True,
                                 dropout=True,
                                 logging=self.logging))


    def predict(self):
        return tf.nn.softmax(self.outputs)

    def weights(self):
        #alpha1 = self.layers[0].weights
        alpha2 = self.layers[2].weights

        #return tf.stack([alpha1, alpha2], axis=0)
        return alpha2