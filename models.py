from collections import namedtuple

import tensorflow as tf
import math
import utils

#import layers as layers

from utils import BipartiteEdgePredLayer
from aggregators import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver()
        save_path = saver.save(sess, self.save_dir+"%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver()
        save_path = self.save_dir + "%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GeneralizedModel(Model):
    """
    Base class for models that aren't constructed from traditional, sequential layers.
    Subclasses must set self.outputs in _build method

    (Removes the layers idiom from build method of the Model class)
    """

    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)
        

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

# SAGEInfo is a namedtuple that specifies the parameters 
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'output_dim' # the output (i.e., hidden) dimension
    ])

class SampleAndAggregate(GeneralizedModel):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, placeholders, features, node_type, node_type_dict, node_type_size, adj, degrees, degrees_B,
            layer_infos,layer_infos_A, layer_infos_B, concat=True, aggregator_type="mean", 
            model_size="small", loss_type="xent", identity_dim=0, num_heads=1, save_dir="tmp",
            **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features. 
                        NOTE: Pass a None object to train in featureless mode (identity features for nodes)!
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - identity_dim: Set to positive int to use identity features (slow and cannot generalize, but better accuracy)
        '''
        super(SampleAndAggregate, self).__init__(**kwargs)
        if aggregator_type == "mean":
            print("You have choose mean aggregator")
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "gat_multi":
            self.aggregator_cls = MultiAttentionAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.inputs_A = placeholders["batch_A"]
        self.inputs_B = placeholders["batch_B"]

        #self.batch_layer = placeholders["batch_layer"]
        self.model_size = model_size
        self.num_heads = num_heads
        self.loss_type = loss_type
        self.save_dir = save_dir
        self.adj_info = adj
        self.node_type_embeds = tf.get_variable("node_type_embeddings", [len(node_type_dict), node_type_size])
        #features = None
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.node_type_emb = tf.nn.embedding_lookup(self.node_type_embeds, node_type)
        self.degrees = degrees
        self.degrees_B = degrees_B
        self.concat = concat

        self.dims = [(0 if features is None else features.shape[1]) + (identity_dim + (2*node_type_size))]
        self.dims_A = [(0 if features is None else features.shape[1]) + (identity_dim + (2*node_type_size))]
        self.dims_B = [(0 if features is None else features.shape[1]) + (identity_dim + (2*node_type_size))]

        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.dims_A.extend([layer_infos[i].output_dim for i in range(len(layer_infos_A))])
        self.dims_B.extend([layer_infos[i].output_dim for i in range(len(layer_infos_B))])

        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos
        self.layer_infos_A = layer_infos_A
        self.layer_infos_B = layer_infos_B

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        with tf.device('/gpu:0'):
            self.build()

    def sample(self, inputs, layer_infos, batch_size=None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        
        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler
            node = sampler((samples[k], layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * batch_size,]))
            support_sizes.append(support_size)
        return samples, support_sizes


    def aggregate(self, samples, adj_samples, node_type_feature, input_features, dims, num_samples, support_sizes, batch_size=None,
            aggregators=None, name=None, concat=False, model_size="small",num_heads=1):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """

        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        hidden = []
        for i in range(0,len(samples)):
            node_fea = tf.nn.embedding_lookup(input_features, samples[i])
            node_type = tf.nn.embedding_lookup(node_type_feature, samples[i])
            adj_node_type = tf.nn.embedding_lookup(node_type_feature, adj_samples[i])
            edge_type = tf.add(node_type, adj_node_type)
            node_fea = tf.concat([node_fea, adj_node_type, edge_type], axis=1)
            hidden.append(node_fea)
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                            dropout=self.placeholders['dropout'],
                            name=name, concat=concat, model_size=model_size, num_heads=num_heads,sample_num=num_samples[layer])
                else:
                    aggregator2 = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                            dropout=self.placeholders['dropout'],
                            name=name, concat=concat, model_size=model_size, num_heads=num_heads,sample_num=num_samples[layer+1])
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                            dropout=self.placeholders['dropout'],
                            name=name, concat=concat, model_size=model_size, num_heads=num_heads,sample_num=num_samples[layer])
                aggregators.append(aggregator)
                aggregators.append(aggregator2)
            else:
                if layer == 0:
                    aggregator = aggregators[layer]
                    aggregator2 = aggregators[layer +1]
                else:
                    aggregator = aggregators[layer+1]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples) - hop - 1],
                              dim_mult*dims[layer]]
                if layer == 0 and hop == 0:
                    h = aggregator2((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                else:
                    h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators

    def _build(self):
        labels = tf.reshape(
                tf.cast(self.placeholders['batch2'], dtype=tf.int64),
                [self.batch_size, 1])
        labels_B = tf.reshape(
                tf.cast(self.placeholders['batch_B'], dtype=tf.int64),
                [self.batch_size, 1])
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))
        self.neg_samples_B, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_B,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=False,
            range_max=len(self.degrees_B),
            distortion=0.75,
            unigrams=self.degrees_B.tolist()))
           
        # perform "convolution"
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos)
        samples_A, support_sizes_A = self.sample(self.inputs_A, self.layer_infos_A)
        samples_B, support_sizes_B = self.sample(self.inputs_B, self.layer_infos_B)
        samples_B_subgraph, support_sizes_B_subgraph = self.sample(self.inputs_B, self.layer_infos_B)

        #samples_layer, support_layers = self.sample(self.batch_layer,self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        num_samples_A = [layer_info.num_samples for layer_info in self.layer_infos_A]
        num_samples_B = [layer_info.num_samples for layer_info in self.layer_infos_B]
        self.outputs1, self.aggregators = self.aggregate(samples1, samples2, [self.node_type_emb], [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size, num_heads=self.num_heads)
        self.outputs2, _ = self.aggregate(samples2, samples1, [self.node_type_emb], [self.features], self.dims, num_samples,
                support_sizes2, aggregators=self.aggregators, concat=self.concat,model_size=self.model_size, num_heads=self.num_heads)

        self.outputs_A, _ = self.aggregate(samples_A, samples_B, [self.node_type_emb], [self.features], self.dims_A, num_samples,
                support_sizes_A, aggregators=self.aggregators, concat=self.concat,model_size=self.model_size, num_heads=self.num_heads)
        self.outputs_B, _ = self.aggregate(samples_B, samples_A, [self.node_type_emb], [self.features], self.dims_B, num_samples,
                support_sizes_B, aggregators=self.aggregators, concat=self.concat,model_size=self.model_size, num_heads=self.num_heads)
        self.outputs_B_subgraph, _ = self.aggregate(samples_B_subgraph, samples_B, [self.node_type_emb], [self.features], self.dims_B, num_samples,
                support_sizes_B_subgraph, aggregators=self.aggregators, concat=self.concat,model_size=self.model_size, num_heads=self.num_heads)


        neg_samples, neg_support_sizes = self.sample(self.neg_samples, self.layer_infos, FLAGS.neg_sample_size)
        neg_samples_B, neg_support_sizes_B = self.sample(self.neg_samples_B, self.layer_infos_B, FLAGS.neg_sample_size)
        self.neg_outputs, _ = self.aggregate(neg_samples, neg_samples, [self.node_type_emb], [self.features], self.dims, num_samples,
                neg_support_sizes, batch_size=FLAGS.neg_sample_size, aggregators=self.aggregators,
                concat=self.concat, model_size=self.model_size, num_heads=self.num_heads)
        self.neg_outputs_B, _ = self.aggregate(neg_samples_B, neg_samples_B, [self.node_type_emb], [self.features], self.dims_B, num_samples,
                neg_support_sizes_B, batch_size=FLAGS.neg_sample_size, aggregators=self.aggregators,
                concat=self.concat, model_size=self.model_size, num_heads=self.num_heads)

        dim_mult = 2 if self.concat else 1
        self.link_pred_layer = BipartiteEdgePredLayer(dim_mult*self.dims[-1],
                dim_mult*self.dims[-1], self.placeholders, act=tf.nn.sigmoid, 
                loss_fn=self.loss_type, bilinear_weights=False,
                name='edge_predict')

    def build(self):
        with tf.device('/gpu:0'):
            with tf.variable_scope("aggregation"):
                self._build()

            # TF graph management
            self._loss()
            self._accuracy()
            self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                    for grad, var in grads_and_vars]
            self.grad, _ = clipped_grads_and_vars[0]
            self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.neg_outputs)
        self.loss += self.link_pred_layer.loss(self.outputs_A, self.outputs_B, self.neg_outputs_B)
        self.loss += self.link_pred_layer.loss(self.outputs_B, self.outputs_B_subgraph, self.neg_outputs_B)
        #self.loss += self.link_pred_layer.loss(self.score_student, self.score_teacher, self.score_negative)
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
        # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
        self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)
