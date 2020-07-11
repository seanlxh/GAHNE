
from GAHNE.inits import *
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
        self.weights = None

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):  # name = 'graphconvolution_id'
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
#
# class GraphAttention(Layer):
#     """Graph convolution layer."""
#     def __init__(self, input_dim, output_dim, placeholders, indices, total_num,dropout=0.,
#                  sparse_inputs=False, act=tf.nn.tanh, bias=False,
#                  featureless=False, **kwargs):
#         super(GraphAttention, self).__init__(**kwargs)
#
#         if dropout:
#             self.dropout = placeholders['dropout']
#         else:
#             self.dropout = 0.
#
#         self.act = act
#         self.support = placeholders['support']
#         self.sparse_inputs = sparse_inputs
#         self.featureless = featureless
#         self.output_dim = output_dim
#         self.total_num = total_num
#         self.bias = bias
#         self.indices = indices
#         self.input_dim = input_dim
#         self.one_input_dim = int(self.input_dim / len(self.support))
#         # helper variable for sparse dropout
#         self.num_features_nonzero = placeholders['num_features_nonzero']
#
#         with tf.variable_scope(self.name + '_vars'):
#
#             self.vars['weights_w'] = glorot([self.one_input_dim, output_dim],
#                                                         name='weights_w')
#             if self.bias:
#                 self.vars['bias'] = zeros([output_dim], name='bias')
#
#             self.vars['weights_q'] = ones([output_dim, 1],
#                                                     name='weights_q')
#
#         if self.logging:
#             self._log_vars()
#
#     def _call(self, inputs):
#         x = inputs
#
#         # dropout
#         if self.sparse_inputs:
#             x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
#         else:
#             x = tf.nn.dropout(x, 1-self.dropout)
#
#         # convolve
#         supports = list()
#         for i in range(len(self.support)):
#             if not self.featureless:
#                 tmp_x = x[:, i * self.one_input_dim : (i + 1) * self.one_input_dim]
#                 cur_x = tf.gather(tmp_x, tf.Variable(self.indices[i]))
#                 pre_sup = dot(cur_x, self.vars['weights_w'],
#                               sparse=self.sparse_inputs)
#             else:
#                 pre_sup = self.vars['weights_w']
#
#
#             if self.bias:
#                 pre_sup += self.vars['bias']
#             pre_sup = self.act(pre_sup)
#
#             sup = dot(pre_sup,
#                  self.vars['weights_q'], sparse=self.sparse_inputs)
#
#             support = tf.reduce_sum(sup) /  tf.cast(tf.shape(sup)[0],dtype=tf.float32)
#
#             supports.append(support)
#
#         output = tf.stack(supports, axis=0)
#
#         args = tf.nn.softmax(output)
#
#         final = zeros([self.total_num, self.one_input_dim], name='final')
#
#         for i in range(len(self.support)):
#             tmp_x = x[:, i * self.one_input_dim: (i + 1) * self.one_input_dim]
#             final = final + tmp_x * args[i]
#
#         return final

class GraphTraditionalConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphTraditionalConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support_tradition']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

            # self.vars['tmp'] = zeros([input_dim, output_dim], name='tmp')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']
        # return dot(x, self.vars['tmp'],
        #                       sparse=self.sparse_inputs)
        return self.act(output)


class GraphQWTAttention(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphQWTAttention, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_atten'] = glorot([output_dim, 128],
                                                    name='weights_atten')
            self.vars['weights_query'] = tf.Variable(tf.random_normal([128], stddev=0.1))

            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
                self.vars['bias_atten1'] = zeros([128], name='bias_atten1')

        if self.logging:
            self._log_vars()

    def Relu(self, x):
        return tf.nn.relu(x)

    def Tanh(self, x):
        return tf.nn.tanh(x)

    def Sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)

        input_x = tf.stack(supports, 2)  #[num][hidden_size][channel]
        input_x = tf.transpose(input_x, [2, 0, 1])  #[channel][num][hidden_size]
        excitation = tf.tensordot(input_x, self.vars['weights_atten'], axes=1) #[channel][num][qeury_size]
        # excitation += self.vars['bias_atten1']
        excitation = self.Sigmoid(excitation)
        excitation = tf.tensordot(excitation, self.vars['weights_query'], axes=1)#[channel][num]
        num = tf.cast(tf.shape(excitation), dtype=tf.float32)
        excitation = tf.reduce_sum(excitation, 1) / num[1]
        alphas = tf.nn.softmax(excitation, name='alphas')         #[channel][num]
        self.weights = alphas
        alphas = tf.expand_dims(alphas, -1)
        alphas = tf.expand_dims(alphas, -1)
        ########################
        output = tf.reduce_sum(input_x * alphas, 0)
        # # bias
        if self.bias:
            output += self.vars['bias']
        print(output)
        return self.act(output)

class GraphGate(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphGate, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):

            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))

                self.vars['weights_gate' + str(i)] = glorot([output_dim, output_dim],
                                                    name='weights_atten' + str(i))

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def Relu(self, x):
        return tf.nn.relu(x)

    def Tanh(self, x):
        return tf.nn.tanh(x)

    def Sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        excitations = list()
        results = list()

        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)

            supports.append(support)
            excitation = dot(support, self.vars['weights_gate' + str(i)])
            excitation = self.Sigmoid(excitation)
            excitations.append(excitation)
            result = excitation * support
            results.append(result)

        output = tf.reduce_sum(results, 0)
        print("output")
        print(output)
        # # bias
        if self.bias:
            output += self.vars['bias']

        #output = tf.concat([output, x], 1)
        # output = tf.add(output, x)
        print(output)
        return self.act(output)
        #return output


class GraphWeight(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphWeight, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):

            self.vars['weights_gate'] = glorot([output_dim, output_dim],
                                                        name='weights_atten')
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))

            if self.bias:
                self.vars['bias_gate'] = zeros([output_dim], name='bias_gate')
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def Relu(self, x):
        return tf.nn.relu(x)

    def Tanh(self, x):
        return tf.nn.tanh(x)

    def Sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        excitations = list()
        results = list()

        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)

            supports.append(support)
        #     excitation = dot(support, self.vars['weights_gate'])
        #     if self.bias:
        #         excitation += self.vars['bias_gate']
        #     excitation = self.Sigmoid(excitation)
        #     excitations.append(excitation)
        # output = tf.reduce_mean(excitations, 0)
        output = tf.reduce_mean(supports, 0)

        print("output")
        print(output)
        # # bias
        if self.bias:
            output += self.vars['bias']

        #output = tf.concat([output, x], 1)
        # output = tf.add(output, x)
        print(output)
        return self.act(output)
        #return output



class GraphQWTAttentionwithK(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, K=1, **kwargs):
        super(GraphQWTAttentionwithK, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim
        self.K = K
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_atten'] = glorot([output_dim, 128],
                                                    name='weights_atten')
            self.vars['weights_query'] = tf.Variable(tf.random_normal([128], stddev=0.1))

            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
                self.vars['bias_atten1'] = zeros([128], name='bias_atten1')

        if self.logging:
            self._log_vars()

    def Relu(self, x):
        return tf.nn.relu(x)

    def Tanh(self, x):
        return tf.nn.tanh(x)

    def Sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            tmp_sup = []
            support = dot(self.support[i], pre_sup, sparse=True)
            tmp_sup.append(support)
            for uu in range(self.K-1):
                support = dot(self.support[i], support, sparse=True)
                tmp_sup.append(support)
            final = tf.reduce_sum(tmp_sup, 0)
            print(final)
            supports.append(final)

        input_x = tf.stack(supports, 2)  #[num][hidden_size][channel]
        print("input_x1")
        print(input_x)
        input_x = tf.transpose(input_x, [2, 0, 1])  #[channel][num][hidden_size]
        print("input_x2")
        print(input_x)
        excitation = tf.tensordot(input_x, self.vars['weights_atten'], axes=1) #[channel][num][qeury_size]
        # excitation += self.vars['bias_atten1']
        excitation = self.Sigmoid(excitation)
        print("excitation1")
        print(excitation)
        excitation = tf.tensordot(excitation, self.vars['weights_query'], axes=1)#[channel][num]
        # excitation += self.vars['bias_atten2']
        print("excitation2")
        print(excitation)
        num = tf.cast(tf.shape(excitation), dtype=tf.float32)

        print(num[1])
        excitation = tf.reduce_sum(excitation, 1) / num[1]
        print("excitation3")
        print(excitation)
        alphas = tf.nn.softmax(excitation, name='alphas')         #[channel][num]
        print("alphas1")
        print(alphas)
        self.weights = alphas
        ########################
        # for i in range(len(supports)):
        #     supports[i] = supports[i] * alphas[i]
        ########################
        alphas = tf.expand_dims(alphas, -1)
        alphas = tf.expand_dims(alphas, -1)
        print("alphas2")
        print(alphas)
        ########################
        output = tf.reduce_sum(input_x * alphas, 0)
        ########################
        # output = tf.concat(supports, axis=1)
        ########################

        print("output")
        print(output)
        # # bias
        if self.bias:
            output += self.vars['bias']

        #output = tf.concat([output, x], 1)
        # output = tf.add(output, x)
        print(output)
        return self.act(output)
        #return output