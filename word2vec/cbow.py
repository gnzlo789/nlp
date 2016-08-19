# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
%matplotlib inline
from __future__ import print_function
#import words2vec.utils
import collections
import math
import numpy as np
import random
import tensorflow as tf
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
#from sklearn.manifold import TSNE

batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.

# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 32 # Number of negative examples to sample.

# General defines
context_window = 2 * skip_window
num_labels = batch_size / context_window

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels  = tf.placeholder(tf.int32, shape=[num_labels, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    # W . Vword + bi
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],
                                               -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal(
                    [vocabulary_size,embedding_size],
                    stddev=1.0 / math.sqrt(embedding_size))
    )

    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)

    # seq_ids only needs to be generated once so do this as a numpy array rather than a tensor.
    seq_ids = np.zeros(batch_size, dtype=np.int32)
    cur_id = -1
    for i in range(batch_size):
        if i % context_window == 0:
            cur_id = cur_id + 1
        seq_ids[i] = cur_id
    print(seq_ids)

    # use segment_sum to add together the related words and reduce the output to be num_labels in size.
    final_embed = tf.segment_sum(embed, seq_ids)

    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        # tf.nn.nce_loss(same parameters)
        tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, final_embed,
                                   train_labels, num_sampled, vocabulary_size)
    )

    # Optimizer.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    #     A . B
    # ------------- = Distance
    # ||A|| . ||B||
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


num_steps = 100001

with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    print('Initialized')

    average_loss = 0

    for step in range(num_steps):
        batch_data, batch_labels = generate_batch_cbow(batch_size, skip_window)
        feed_dict = {train_dataset: batch_data,
                     train_labels: batch_labels}

        _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0

        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 20000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)

    final_embeddings = normalized_embeddings.eval()
    # saver = tf.train.Saver()
    # saver.save(session, 'cbow_word2vec', global_step = 0)

print(final_embeddings[0])

# t-SNE plot
num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])

def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15,15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2),
                       textcoords='offset points',ha='right', va='bottom')
    pylab.show()

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)
