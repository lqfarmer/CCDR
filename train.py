from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import time

from gat.models import SampleAndAggregate, SAGEInfo
from gat.utils import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'sage_mean', 'model names. sage_mean(basic graphsage)/gat_multi/gat_self/sage_seq')
flags.DEFINE_float('learning_rate', 0.00001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'name of the object file that stores the training data. must be specified.')
flags.DEFINE_string('adj_name', 'adj', 'name of training data. must be specified. format: node adj_node1:adj_node2...')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')
flags.DEFINE_integer('samples_1_s', 10, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2_s', 5, 'number of users samples in layer 2')
flags.DEFINE_integer('dim_1_s', 50, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2_s', 50, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_1', 50, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 50, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_boolean('layer_sample', False, 'sample positive samples by layer')
flags.DEFINE_integer('neg_sample_size', 30, 'number of negative samples')
flags.DEFINE_integer('pos_sample_size', 10, 'number of positive samples')
flags.DEFINE_integer('batch_size', 1024, 'minibatch size.')
flags.DEFINE_integer('num_heads', 10, 'num of heads for multi_head attention, must divide feature size to Zero.')
flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')
flags.DEFINE_integer('identity_dim', 50, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_integer('node_type_size', 5, 'Set to positive value to use node type feature of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_string('loss_type', 'xent', 'loss type : xent / skipgram / hinge')
flags.DEFINE_string('sample_mode', 'node', 'edge pair sample mode : node or edge')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('save_per_epoch', 10, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 512, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")
flags.DEFINE_string('result_type', 'emb', 'final result type: emb or npy')
flags.DEFINE_string('sample_choice', 'random', 'final result type: emb or npy')

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def log_dir():
    log_dir = FLAGS.base_log_dir + "/model_output"
    log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

def incremental_evaluate(sess, model, minibatch_iter, size):
    t_test = time.time()
    finished = False
    val_losses = []
    val_mrrs = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                            feed_dict=feed_dict_val)
        val_losses.append(outs_val[0])
        val_mrrs.append(outs_val[2])
    return np.mean(val_losses), np.mean(val_mrrs), (time.time() - t_test)

def save_val_embeddings(sess, model, minibatch_iter, size, out_dir, idx2id, result_type="emb", mod=""):
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    name = "val"
    while not finished:
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.mrr, model.outputs1], 
                            feed_dict=feed_dict_val)
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i,:])
                nodes.append(edge[0])
                seen.add(edge[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    val_embeddings = np.vstack(val_embeddings)
    if result_type == "npy":
        np.save(out_dir + name + mod + ".npy",  val_embeddings)
        with open(out_dir + name + mod + ".txt", "w") as fp:
            for i in range(0,len(nodes)):
                node = nodes[i]
                if node in idx2id:
                    fp.write(idx2id[node] + "\t" + str(node) + "\n")
    elif result_type == "emb":
        with open(out_dir + "final_node_hidden", "w") as fp:
            for i in range(0,len(nodes)):
                if nodes[i] in idx2id:
                    temp = []
                    for emb in val_embeddings[i]:
                        temp.append(str(emb))
                    if len(temp) > 0:
                        fp.write(idx2id[nodes[i]] + "\t" + ":".join(temp) + "\n")

def construct_placeholders():
    # Define placeholders FLAGS.batch_size
    placeholders = {
        'batch1' : tf.placeholder(tf.int32, shape=(None,), name='batch1'),
        'batch2' : tf.placeholder(tf.int32, shape=(None,), name='batch2'),
        'batch_A' : tf.placeholder(tf.int32, shape=(None,), name='batch_A'),
        'batch_B' : tf.placeholder(tf.int32, shape=(None,), name='batch_B'),
        # negative samples for all nodes in the batch
        'neg_samples': tf.placeholder(tf.int32, shape=(None,),
            name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def train(train_data=None, test_data=None):

    save_dir = log_dir()
    context_pairs = None
    placeholders = construct_placeholders()
    minibatch = EdgeMinibatchIterator(
            FLAGS.train_prefix,
            placeholders, batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree,
            pos_num=FLAGS.pos_sample_size,
            num_neg_samples=FLAGS.neg_sample_size,
            context_pairs = context_pairs,
            adj_name=FLAGS.adj_name,
            layer_sample=FLAGS.layer_sample,
            sample_choice=FLAGS.sample_choice)
    features = None
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape, name='adj_ph')
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    adj_info_ph_A = tf.placeholder(tf.int32, shape=minibatch.adj_A.shape, name='adj_ph_A')
    adj_info_A = tf.Variable(adj_info_ph_A, trainable=False, name="adj_info_A")
    adj_info_ph_B = tf.placeholder(tf.int32, shape=minibatch.adj_B.shape, name='adj_ph_B')
    adj_info_B = tf.Variable(adj_info_ph_B, trainable=False, name="adj_info_B")

    if FLAGS.model == 'sage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        sampler_A = UniformNeighborSampler(adj_info_A)
        sampler_B = UniformNeighborSampler(adj_info_B)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        layer_infos_A = [SAGEInfo("node", sampler_A, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler_A, FLAGS.samples_2, FLAGS.dim_2)]
        layer_infos_B = [SAGEInfo("node", sampler_B, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler_B, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     minibatch.node_type,
                                     minibatch.node_type_dict,
                                     FLAGS.node_type_size,
                                     adj_info,
                                     minibatch.deg,
                                     minibatch.deg_B,
                                     layer_infos=layer_infos,
                                     layer_infos_A=layer_infos_A,
                                     layer_infos_B=layer_infos_B,
                                     model_size=FLAGS.model_size,
                                     identity_dim=FLAGS.identity_dim,
                                     loss_type=FLAGS.loss_type,
                                     save_dir=save_dir,
                                     logging=True)
    elif FLAGS.model == 'gat_multi':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        sampler_A = UniformNeighborSampler(adj_info_A)
        sampler_B = UniformNeighborSampler(adj_info_B)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        layer_infos_A = [SAGEInfo("node", sampler_A, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler_A, FLAGS.samples_2, FLAGS.dim_2)]
        layer_infos_B = [SAGEInfo("node", sampler_B, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler_B, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                     features,
                                     minibatch.node_type,
                                     minibatch.node_type_dict,
                                     FLAGS.node_type_size,
                                     adj_info,
                                     minibatch.deg,
                                     minibatch.deg_B,
                                     layer_infos=layer_infos,
                                     layer_infos_A=layer_infos_A,
                                     layer_infos_B=layer_infos_B,
                                     concat=False,
                                     aggregator_type="gat_multi",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     num_heads = FLAGS.num_heads,
                                     save_dir=save_dir,
                                     logging=True)

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    print_variable_summary() 
    # Initialize session
    sess = tf.Session(config=config)
    checkpoint_path = os.path.join(os.getcwd(),save_dir[2:],"checkpoint")
    if os.path.exists(checkpoint_path):
        print("load latest model")
        model.load(sess=sess)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
     
    # Init variables
    fd = {}
    fd.update({adj_info_ph_A: minibatch.adj_A})
    fd.update({adj_info_ph_B: minibatch.adj_B})
    fd.update({adj_info_ph: minibatch.adj})
    sess.run(tf.global_variables_initializer(), feed_dict=fd)
    # Train model
    
    train_shadow_mrr = None
    shadow_mrr = None

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    #train_adj_info = tf.assign(adj_info, minibatch.adj)
    #val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    print("Start traing with sample mode : %s"%(FLAGS.sample_mode))
    for epoch in range(FLAGS.epochs):
        minibatch.shuffle(mode=FLAGS.sample_mode) 
        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        print("iteration per epoch : %d" % (minibatch.iteration_per_epoch(mode=FLAGS.sample_mode)))
        epoch_val_costs.append(0)
        while not minibatch.end(mode=FLAGS.sample_mode):
            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict(mode=FLAGS.sample_mode)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all, 
                    model.mrr, model.outputs1], feed_dict=feed_dict)
            train_cost = outs[2]
            train_mrr = outs[5]
            if train_shadow_mrr is None:
                train_shadow_mrr = train_mrr#
            else:
                train_shadow_mrr -= (1-0.99) * (train_shadow_mrr - train_mrr)

            if iter % FLAGS.validate_iter == 0:
                # Validation
                #sess.run(val_adj_info.op)
                val_cost, ranks, val_mrr, duration  = evaluate(sess, model, minibatch, size=FLAGS.validate_batch_size)
                #sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost
            if shadow_mrr is None:
                shadow_mrr = val_mrr
            else:
                shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)
    
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                print("Iter:", '%04d' % iter, 
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_mrr=", "{:.5f}".format(train_mrr), 
                      "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr), # exponential moving average
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_mrr=", "{:.5f}".format(val_mrr), 
                      "val_mrr_ema=", "{:.5f}".format(shadow_mrr), # exponential moving average
                      "time=", "{:.5f}".format(avg_time))

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break
        if epoch % FLAGS.save_per_epoch == 0:
            model.save(sess=sess)
        if total_steps > FLAGS.max_total_steps:
                break
    model.save(sess=sess) 
    print("Optimization Finished!")
    print("Save node embedding")
    node_embs_temp = []
    node_embs = sess.run(model.embeds)
    node_embeddings = np.vstack(node_embs)
    np.save(save_dir + "node_embs.npy",  node_embeddings)

    print("Save final hidden state")
    if FLAGS.save_embeddings:
        #sess.run(val_adj_info.op)
        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir(), minibatch.idx2id,result_type=FLAGS.result_type)


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
