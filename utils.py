from __future__ import division
from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import tensorflow as tf

WALK_LEN=5
N_WALKS=50

#edge iterator
np.random.seed(123)

def print_variable_summary():
    import pprint
    variables = sorted([[v.name, v.get_shape()] for v in tf.global_variables()])
    pprint.pprint(variables)

class EdgeMinibatchIterator(object):

    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """
    def __init__(self, prefix_dir, placeholders, context_pairs=None, batch_size=100, max_degree=25, pos_num=10,
            layer_sample=False, adj_name="adj", sample_choice="random", **kwargs):

        
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.adj_name = adj_name
        self.sample_choice = sample_choice
        self.batch_num = 0
        self.pos_num = pos_num
        self.layer_sample = layer_sample

        self.nodes, self.id2idx, self.idx2id, self.edges, self.val_edges, self.adj, self.weights, self.node_type, self.node_type_dict, self.adj_A, self.adj_B, self.weights_A, self.weights_B, self.val_edges_cl = self.preload_data(prefix_dir)
        self.dic_adj = self.adj
        self.dic_adj_A = self.adj_A
        self.dic_adj_B = self.adj_B
        #self.nodes = np.random.permutation(self.nodes)
        adj_norm, self.deg, adj_norm_A, self.deg_A, adj_norm_B, self.deg_B = self.construct_adj()
        self.adj = self.test_adj = adj_norm
        self.adj_A = self.test_adj_A = adj_norm_A
        self.adj_B = self.test_adj_B = adj_norm_B
        
        #self.train_edges = self.edges = np.random.permutation(self.edges)
        #self.train_edges = self.edges
        #self.val_set_size = len(self.val_edges)

    def preload_data(self, prefix):
        adj_path = os.path.join(os.getcwd(),prefix[2:], self.adj_name)
        if not os.path.exists(adj_path):
            print("Input file : %s not exists !" %(adj_path))
        else:
            print("load text data from: %s" %(adj_path))

        nodes = []
        node_type = []
        node_type_dict = {}
        id2idx = {}
        idx2id = {}
        adj = {}
        adj_A = {}
        adj_B = {}
        edges = []
        weights = {}
        weights_A = {}
        weights_B = {}
        test_edges = []
        ct = 0
        ct_type = 0
        for l in open(adj_path):
            l = l.strip().split("\t")
            if len(l) != 4: continue
            sor_temp = l[0].split(":#:")
            if len(sor_temp) != 2: continue
            sor_id = sor_temp[0]
            if l[0] in id2idx or len(sor_id) < 1 or len(l[3].split("::")) < 1: continue
            if sor_id not in id2idx:
                id2idx[sor_id] = ct
                idx2id[ct] = sor_id
                ct += 1
            ntype = sor_temp[1]
            if ntype not in node_type_dict:
                node_type_dict[ntype] = ct_type
                ct_type += 1
        for l in open(adj_path):
            ll = l.strip()
            l = l.strip().split("\t")
            if len(l) != 4: continue
            sor_temp = l[0].split(":#:")
            sor_id = sor_temp[0]
            if (not id2idx.has_key(sor_temp[0])) or (not sor_temp[1] in node_type_dict): continue
            if id2idx[sor_id] in adj: continue
            #node_type.append(node_type_dict[sor_temp[1]])
            if len(sor_id) < 1 or len(l[3].split("::")) < 1: continue
            temp = []
            weight_temp = []
            temp_A =  []
            weight_temp_A = []
            temp_B = []
            weight_temp_B = []
            #nodes.append(id2idx[sor_id])
            adj_nodes = l[1].split("::")
            adj_nodes_A = l[2].split("::")
            adj_nodes_B = l[3].split("::")
            end = len(adj_nodes)
            if len(adj_nodes) > 10 and len(test_edges) < 500:
                end = len(adj_nodes) - 3
            for i in range(0,end):
                node_weight = adj_nodes[i].split("#")
                if len(node_weight) < 3 or len(node_weight[0]) < 1 or len(node_weight[1]) < 1 or len(node_weight[2]) < 1: continue
                node = node_weight[0]
                if node in id2idx:
                    temp.append(id2idx[node])
                    weight_temp.append(float(node_weight[2]))
            for j in range(end, len(adj_nodes)):
                node_weight = adj_nodes[i].split("#")
                if len(node_weight) < 3 or len(node_weight[0]) < 1 or len(node_weight[1]) < 1 or len(node_weight[2]) < 1: continue
                node = node_weight[0]
                if node in id2idx:
                    temp.append(id2idx[node])
                    weight_temp.append(float(node_weight[2]))
                    test_edges.append((id2idx[sor_id],id2idx[node]))
            for i in range(0,len(adj_nodes_A)):
                node_weight = adj_nodes_A[i].split("#")
                if len(node_weight) < 3 or len(node_weight[0]) < 1 or len(node_weight[1]) < 1 or len(node_weight[2]) < 1: continue
                node = node_weight[0]
                if node in id2idx:
                    temp_A.append(id2idx[node])
                    weight_temp_A.append(float(node_weight[2]))
            for i in range(0,len(adj_nodes_B)):
                node_weight = adj_nodes_B[i].split("#")
                if len(node_weight) < 3 or len(node_weight[0]) < 1 or len(node_weight[1]) < 1 or len(node_weight[2]) < 1: continue
                node = node_weight[0]
                if node in id2idx:
                    temp_B.append(id2idx[node])
                    weight_temp_B.append(float(node_weight[2]))
            if len(temp) < 1 or len(temp_A) < 1 or len(temp_B) < 1:
                continue
            adj[id2idx[sor_id]] =  temp
            weights[id2idx[sor_id]] = weight_temp
            adj_A[id2idx[sor_id]] =  temp_A
            weights_A[id2idx[sor_id]] = weight_temp_A
            adj_B[id2idx[sor_id]] =  temp_B
            weights_B[id2idx[sor_id]] = weight_temp_B
            node_type.append(node_type_dict[sor_temp[1]])
            nodes.append(id2idx[sor_id])

        test_edges_filter = []
        for node1, node2 in test_edges:
            if node1 not in adj or node2 not in adj: continue
            test_edges_filter.append((node1,node2))
        
        return nodes, id2idx, idx2id,  edges, test_edges_filter, adj, weights, node_type, node_type_dict, adj_A, adj_B, weights_A, weights_B, test_edges_filter

    def construct_adj(self):
        #adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        adj = len(self.adj)*np.ones((len(self.adj), self.max_degree))
        deg = np.zeros((len(self.adj),))
        adj_A = len(self.adj_A)*np.ones((len(self.adj_A), self.max_degree))
        #adj_A = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg_A = np.zeros((len(self.adj),))
        #adj_B = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        adj_B = len(self.adj_B)*np.ones((len(self.adj_B), self.max_degree))
        deg_B = np.zeros((len(self.adj),))


        for nodeid in self.nodes:
            if not nodeid in self.adj or not nodeid in self.adj_A or not nodeid in self.adj_B: continue
            neighbors = np.array(self.adj[nodeid])
            neighbors_A = np.array(self.adj_A[nodeid])
            neighbors_B = np.array(self.adj_B[nodeid])

            if len(neighbors) == 0 or len(neighbors_A) == 0 or len(neighbors_B) == 0:
                continue
            deg[nodeid] = len(neighbors)
            deg_A[nodeid] = len(neighbors_A)
            deg_B[nodeid] = len(neighbors_B)

            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors

            if len(neighbors_A) > self.max_degree:
                neighbors_A = np.random.choice(neighbors_A, self.max_degree, replace=False)
            elif len(neighbors_A) < self.max_degree:
                neighbors_A = np.random.choice(neighbors_A, self.max_degree, replace=True)
            adj_A[nodeid, :] = neighbors_A
            if len(neighbors_B) > self.max_degree:
                neighbors_B = np.random.choice(neighbors_B, self.max_degree, replace=False)
            elif len(neighbors_B) < self.max_degree:
                neighbors_B = np.random.choice(neighbors_B, self.max_degree, replace=True)
            adj_B[nodeid, :] = neighbors_B

        print("total length of dic_adj : %d"%(len(self.adj)))
        return adj, deg, adj_A, deg_A, adj_B, deg_B
    def iteration_per_epoch(self, mode="edge"):
        return len(self.nodes)/self.batch_size

    def end(self, mode="edge"):
        return self.batch_num * self.batch_size >= len(self.nodes)

    def weighted_choice(self, weights):
        rnd = random.random() * sum(weights)
        for i, w in enumerate(weights):
            rnd -= w
            if rnd < 0:
                return i

    def batch_feed_dict(self, batch_edges, batch_edges_cl, train=True):
        batch1 = []
        batch2 = []
        batch_A = []
        batch_B = []
        for node1, node2 in batch_edges:
            batch1.append(node1)
            batch2.append(node2)
        for node1, node2 in batch_edges_cl:
            batch_A.append(node1)
            batch_B.append(node2)

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})
        feed_dict.update({self.placeholders['batch_A']: batch_A})
        feed_dict.update({self.placeholders['batch_B']: batch_B})

        return feed_dict

    def next_minibatch_feed_dict(self, mode="edge"):
        batch_edges, batch_edges_cl = self.generate_edges_by_nodes()
        return self.batch_feed_dict(batch_edges, batch_edges_cl, train=self.layer_sample)

    def generate_edges_by_nodes(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.nodes))
        batch_edges = []
        batch_edges_cl = []
        for i in range(start_idx,end_idx):
            if not self.nodes[i] in self.dic_adj or (len(self.dic_adj[self.nodes[i]]) < 1) or (not self.nodes[i] in self.weights) or not self.nodes[i] in self.dic_adj_A or not self.nodes[i] in self.dic_adj_B or (not self.nodes[i] in self.weights_A) or (not self.nodes[i] in self.weights_B): continue
            adj_temp = self.dic_adj[self.nodes[i]]
            adj_temp_A = self.dic_adj_A[self.nodes[i]]
            adj_temp_B = self.dic_adj_B[self.nodes[i]]
            if self.sample_choice == "random":
                adj_node = np.random.choice(np.array(adj_temp), 1, replace=False)
                batch_edges.append((self.nodes[i],adj_node[0]))
                adj_node_A = np.random.choice(np.array(adj_temp_A), 1, replace=False)
                adj_node_B = np.random.choice(np.array(adj_temp_B), 1, replace=False)
                batch_edges_cl.append((adj_node_A[0],adj_node_B[0]))
            elif self.sample_choice == "weight_sample":
                weight = self.weights[self.nodes[i]]
                node_idx = self.weighted_choice(weight)
                batch_edges.append((self.nodes[i],adj_temp[node_idx]))
                weight_A = self.weights_A[self.nodes[i]]
                node_idx_A = self.weighted_choice(weight_A)
                weight_B = self.weights_B[self.nodes[i]]
                node_idx_B = self.weighted_choice(weight_B)
                batch_edges_cl.append((adj_temp_A[node_idx_A],adj_temp_B[node_idx_B]))
        return batch_edges, batch_edges_cl
        
    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        edge_list = self.val_edges
        edge_list_cl = self.val_edges_cl
        if size is None:
            return self.batch_feed_dict(edge_list, edge_list_cl, train=False)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            val_edges_cl = [edge_list_cl[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges, val_edges_cl, train=False)

    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size,
            len(edge_list))]
        return self.batch_feed_dict(val_edges, train=False), (iter_num+1)*size >= len(self.val_edges), val_edges

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size,
            len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        return self.batch_feed_dict(val_edges, train=False), (iter_num+1)*size >= len(node_list), val_edges

    def shuffle(self, mode="edge"):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        if mode == "edge":
            self.train_edges = np.random.permutation(self.train_edges)
        else:
            self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0

#basic layer part
# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
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

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

#edge predict part
class BipartiteEdgePredLayer(Layer):
    def __init__(self, input_dim1, input_dim2, placeholders, dropout=False, act=tf.nn.sigmoid,
            loss_fn='xent', neg_sample_weights=1.0,
            bias=False, bilinear_weights=False, **kwargs):
        """
        Basic class that applies skip-gram-like loss
        (i.e., dot product of node+target and node and negative samples)
        Args:
            bilinear_weights: use a bilinear weight for affinity calculation: u^T A v. If set to
                false, it is assumed that input dimensions are the same and the affinity will be 
                based on dot product.
        """
        super(BipartiteEdgePredLayer, self).__init__(**kwargs)
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.act = act
        self.bias = bias
        self.eps = 1e-7

        # Margin for hinge loss
        self.margin = 0.1
        self.neg_sample_weights = neg_sample_weights

        self.bilinear_weights = bilinear_weights

        #if dropout:
        #    self.dropout = placeholders['dropout']
        #else:
        #    self.dropout = 0.
        self.dropout = 0.
        # output a likelihood term
        self.output_dim = 1
        with tf.variable_scope(self.name + '_vars'):
            # bilinear form
            if bilinear_weights:
                #self.vars['weights'] = glorot([input_dim1, input_dim2],
                #                              name='pred_weights')
                self.vars['weights'] = tf.get_variable(
                        'pred_weights',
                        shape=(input_dim1, input_dim2),
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss
        elif loss_fn == 'skipgram':
            self.loss_fn = self._skipgram_loss
        elif loss_fn == 'hinge':
            self.loss_fn = self._hinge_loss

        if self.logging:
            self._log_vars()

    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [batch_size x feature_size].
        """
        # shape: [batch_size, input_dim1]
        if self.bilinear_weights:
            prod = tf.matmul(inputs2, tf.transpose(self.vars['weights']))
            self.prod = prod
            result = tf.reduce_sum(inputs1 * prod, axis=1)
        else:
            result = tf.reduce_sum(inputs1 * inputs2, axis=1)
        return result

    def neg_cost(self, inputs1, neg_samples, hard_neg_samples=None):
        """ For each input in batch, compute the sum of its affinity to negative samples.
        Returns:
            Tensor of shape [batch_size x num_neg_samples]. For each node, a list of affinities to
                negative samples is computed.
        """
        if self.bilinear_weights:
            inputs1 = tf.matmul(inputs1, self.vars['weights'])
        neg_aff = tf.matmul(inputs1, tf.transpose(neg_samples))
        return neg_aff

    def loss(self, inputs1, inputs2, neg_samples):
        """ negative sampling loss.
        Args:
            neg_samples: tensor of shape [num_neg_samples x input_dim2]. Negative samples for all
            inputs in batch inputs1.
        """
        return self.loss_fn(inputs1, inputs2, neg_samples)

    def _xent_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):
        aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
        return loss

    def _skipgram_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):
        aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
        neg_cost = tf.log(tf.reduce_sum(tf.exp(neg_aff), axis=1))
        loss = tf.reduce_sum(aff - neg_cost)
        return loss

    def _hinge_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):
        aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
        diff = tf.nn.relu(tf.subtract(neg_aff, tf.expand_dims(aff, 1) - self.margin), name='diff')
        loss = tf.reduce_sum(diff)
        self.neg_shape = tf.shape(neg_aff)
        return loss

    def weights_norm(self):
        return tf.nn.l2_norm(self.vars['weights'])

#loss, metric part
def masked_logit_cross_entropy(preds, labels, mask):
    """Logit cross-entropy loss with masking."""
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    loss = tf.reduce_sum(loss, axis=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.maximum(tf.reduce_sum(mask), tf.constant([1.]))
    loss *= mask
    return tf.reduce_mean(loss)

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.maximum(tf.reduce_sum(mask), tf.constant([1.]))
    loss *= mask
    return tf.reduce_mean(loss)


def masked_l2(preds, actuals, mask):
    """L2 loss with masking."""
    loss = tf.nn.l2(preds, actuals)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

#neighborhood sample
"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists

#init param part
def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def split_heads(x, num_heads):
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])

def split_last_dimension(x, n):
  old_shape = x.get_shape().dims
  last = old_shape[-1]
  new_shape = old_shape[:-1] + [n] + [last // n if last else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
  ret.set_shape(new_shape)
  return ret

def combine_heads(x):
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))

def combine_last_two_dimensions(x):
  old_shape = x.get_shape().dims
  a, b = old_shape[-2:]
  new_shape = old_shape[:-2] + [a * b if a and b else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
  ret.set_shape(new_shape)
  return ret

def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map

def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

