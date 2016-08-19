import tensorflow as tf
from collections import Counter, defaultdict

class GloVeModel(object):
    # For more information: https://github.com/GradySimon/tensorflow-glove/blob/master/tf_glove.py

    def __init__(self, dimension, vocabulary_size, window_size,
                 alpha=0.75, lr=0.05, x_max=100,
                 min_ocurrence=1, batch_size=256):
        self.dimension = dimension
        self.vocabulary_size = vocabulary_size
        self.window_size = window_size
        self.alpha = alpha
        self.lr = lr
        self.x_max = x_max
        self.min_ocurrence = min_ocurrence
        self.batch_size = batch_size

    def _context_windows(self, region, left_size, right_size):
        for i, word in enumerate(region):
            start_index = i - left_size
            end_index = i + right_size
            left_context = _window(region, start_index, i - 1)
            right_context = _window(region, i + 1, end_index)
            yield (left_context, word, right_context)

    def word_counts(self, corpus):
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        for region in corpus:
            word_counts.update(region)
            for l_context, word, r_context in self._context_windows(region, self.window_size, self.window_size):
                for i, context_word in enumerate(l_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in corpus. Did you try to reuse a generator?")
        self.__words = [word for word, count in word_counts.most_common(vocab_size)
                        if count >= min_occurrences]
        self.__word_to_id = {word: i for i, word in enumerate(self.__words)}
        self.__cooccurrence_matrix = {
            (self.__word_to_id[words[0]], self.__word_to_id[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in self.__word_to_id and words[1] in self.__word_to_id}

    def train(self, num_epochs, log_dir=None, summary_batch_interval=1000,
              tsne_epoch_interval=None):
        should_write_summaries = log_dir is not None and summary_batch_interval
        should_generate_tsne = log_dir is not None and tsne_epoch_interval
        batches = self.__prepare_batches()
        total_steps = 0
        with tf.Session(graph=self.__graph) as session:
            if should_write_summaries:
                summary_writer = tf.train.SummaryWriter(log_dir, graph_def=session.graph_def)
            tf.initialize_all_variables().run()
            for epoch in range(num_epochs):
                shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    i_s, j_s, counts = batch
                    if len(counts) != self.batch_size:
                        continue
                    feed_dict = {
                        self.__focal_input: i_s,
                        self.__context_input: j_s,
                        self.__cooccurrence_count: counts}
                    session.run([self.__optimizer], feed_dict=feed_dict)
                    if should_write_summaries and (total_steps + 1) % summary_batch_interval == 0:
                        summary_str = session.run(self.__summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, total_steps)
                    total_steps += 1
                if should_generate_tsne and (epoch + 1) % tsne_epoch_interval == 0:
                    current_embeddings = self.__combined_embeddings.eval()
                    output_path = os.path.join(log_dir, "epoch{:03d}.png".format(epoch + 1))
                    self.generate_tsne(output_path, embeddings=current_embeddings)
            self.__embeddings = self.__combined_embeddings.eval()
            if should_write_summaries:
                summary_writer.close()


    def build_graph(self):
        self._graph = tf.Graph()
        with self._graph.as_default():
            count_max = tf.constant([self.x_max], dtype=tf.float32,
                                    name='max_coocurrence_count')
            alpha = tf.constant([self.alpha],
                                dtype=tf.float32,
                                name="alpha")
            self._input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                         name="focal_words")
            self._context_input = tf.placeholder(tf.int32,
                                                 shape=[self.batch_size],
                                                 name="context_words")
            # Xij
            self._coocurrence_count = tf.placeholder(tf.float32,
                                                     shape=[self.batch_size],
                                                     name="coocurrence_count")

            # 8: Wi, Wk
            focal_embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.dimension],
                                  -1.0, 1.0),
                name="focal_embeddings")
            context_embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.dimension],
                                  -1.0, 1.0),
                name="context_embeddings")

            focal_embedding = tf.nn.embedding_lookup(
                [focal_embeddings], self._input)
            context_embedding = tf.nn.embedding_lookup(
                [context_embeddings], self._context_input)

            # 8: Terms bi, bk
            focal_biases = tf.Variable(
                tf.random_uniform([self.vocabulary_size], -1.0, 1.0),
                name="focal_biases")
            context_biases = tf.Variable(
                tf.random_uniform([self.vocabulary_size], -1.0, 1.0),
                name="context_biases")

            # It is necesarry to get the word representation, of size d
            focal_bias = tf.nn.embedding_lookup(
                [focal_biases], self._input)
            context_bias = tf.nn.embedding_lookup(
                [context_biases], self._context_input)

            # 9: f(Xij)
            weighting_factor = tf.minimum(
                1.0,
                tf.pow(tf.div(self._coocurrence_count,
                              count_max), self.alpha)
            )

            # 8: transpose(Wi) * Wk. The result dim is d x d, that's why it is
            # necessary to reduce it to size d
            embedding_product = tf.reduce_sum(tf.mul(focal_embedding,
                                                     context_embedding), 1)

            # log(Xik)
            log_cooccurrences = tf.log(tf.to_float(self._coocurrence_count))
            # Term of 8, the final dimension is d
            distance_expr = tf.square(tf.add_n([embedding_product,
                                                focal_bias,
                                                context_bias,
                                                tf.neg(log_cooccurrences)]))

            # Term inside summatory
            single_losses = tf.mul(weighting_factor, distance_expr) # size d
            self.total_loss = tf.reduce_sum(single_losses) # Add every element of the matrix
            tf.scalar_summary("GloVe loss", self.total_loss)

            # Optimizer
            self.optimizer = tf.train.AdagradOptimizer(self.lr).minimize(
                self.total_loss
            )
            self.summary = tf.merge_all_summaries()

            self._combined_embeddings = tf.add(focal_embeddings,
                                               context_embeddings,
                                               name="combined_embeddings")
