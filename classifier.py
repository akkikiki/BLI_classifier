from __future__ import print_function, division

import operator

import numpy as np
import tensorflow as tf

import skipgram


class Classifier(object):
  TRAIN = 'train'
  PREDICT = 'predict'

  def __init__(self, session, vocabulary_s, vocabulary_t, features,
               candidate_generation_train, candidate_generation_pred,
               hidden_dims=None, name='classifier', lr=0.001):
    self._lr = lr
    self._sess = session
    self._words_source = list(vocabulary_s)
    self._words_target = list(vocabulary_t)
    self._word2id_source = {w: i for i, w in enumerate(vocabulary_s)}
    self._word2id_target = {w: i for i, w in enumerate(vocabulary_t)}
    self._feature_extractor = features
    self.graph(hidden_dims, name)
    unitialized_vars = [v for v in tf.global_variables() if not session.run(tf.is_variable_initialized(v))]
    self._sess.run(tf.variables_initializer(unitialized_vars))
    self._loss_value = 0.
    self._candidate_generation_train = candidate_generation_train
    self._candidate_generation_infer = candidate_generation_pred
    self._examples_cache = [None, None]  # keeps last two batches of examples for debugging.

  def graph(self, hidden_dims, name):
    with tf.variable_scope(name):
      self._labels = tf.placeholder(dtype=tf.float32, shape=[None], name='label')
      features = self._feature_extractor.output
      features_infer = self._feature_extractor.output_infer

      with tf.variable_scope('features2logits') as s:
        logits_train = self._get_logits(features, hidden_dims)
        s.reuse_variables()
        logits_infer = self._get_logits(features_infer, hidden_dims)

      self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_train, labels=self._labels))

      optimizer = tf.train.AdamOptimizer(self._lr)
      gvs = optimizer.compute_gradients(self._loss)
      capped_gvs = [(tf.clip_by_norm(grad, 3), var) for grad, var in gvs if grad != None]

      self._train_op = optimizer.apply_gradients(capped_gvs)
      self._probs_infer = tf.nn.sigmoid(logits_infer)

  def _get_logits(self, features, hidden_dims):
    feature_dim = features.get_shape()[1].value
    if hidden_dims:
      dims = [feature_dim] + hidden_dims
      for i in xrange(0, len(dims) - 1):
        d1, d2 = dims[i], dims[i + 1]
        W_h = tf.get_variable('W_h%i' % i, shape=[d1, d2], initializer=tf.random_normal_initializer(stddev=0.01))
        b_h = tf.get_variable('b_h%i' % i, shape=[d2], initializer=tf.zeros_initializer())
        features = tf.sigmoid(tf.matmul(features, W_h) + b_h, 'hidden_layer%i' % i)
      feature_dim = hidden_dims[-1]

    self._W = tf.get_variable('logistic_weights', shape=[feature_dim, 1], initializer=tf.zeros_initializer())
    self._b = tf.get_variable('logistic_biases', shape=[1, 1], initializer=tf.zeros_initializer())

    logits = tf.squeeze(tf.matmul(features, self._W) + self._b)
    return logits

  def input_feeds(self, input, mode=TRAIN):
    feeds = self._feature_extractor.input_feeds(**input)
    if mode == Classifier.TRAIN:
      feeds[self._labels] = input['label']
    return feeds

  def outputs(self, mode=TRAIN):
    if mode == Classifier.TRAIN:
      return [self._train_op, self._loss]
    elif mode == Classifier.PREDICT:
      return self._probs_infer

  def step(self, examples):
    self._examples_cache[1] = self._examples_cache[0]
    self._examples_cache[0] = examples
    examples = self.training_examples(examples)
    if examples['label'].shape[0] == 0:
      return
    input_feeds = self.input_feeds(examples, mode=Classifier.TRAIN)
    outputs = self.outputs()
    _, loss = self._sess.run(outputs, input_feeds)
    self._loss_value = loss

  def training_examples(self, pos_examples):
    source_words, target_words = [], []
    labels = []

    for example in pos_examples:
      # POSITIVE EXAMPLE
      w_s, w_t = example
      if w_s in self._word2id_source and w_t in self._word2id_target:
        source_words.append(w_s)
        target_words.append(w_t)
        labels.append(1)

      # NEGATIVE EXAMPLES
      if w_t in self._word2id_target:
        negatives_source = self._candidate_generation_train.candidates(w_t, False, w_s)
        for w_s_neg in negatives_source:
          source_words.append(w_s_neg)
          target_words.append(w_t)
          labels.append(0)
      if w_s in self._word2id_source:
        negatives_targets = self._candidate_generation_train.candidates(w_s, True, w_t)
        for w_t_neg in negatives_targets:
          source_words.append(w_s)
          target_words.append(w_t_neg)
          labels.append(0)

    batch = {'source/word': source_words,
             'target/word': target_words,
             'label': np.array(labels)}
    return batch

  def loss(self):
    return self._loss_value

  @property
  def loss_variable(self):
    return self._loss

  def last_examples(self):
    return list(self._examples_cache)

  def predict(self, words, source2target=True, threshold=0.5):

    candidates1 = []
    candidates2 = []
    vocab = self._word2id_source if source2target else self._word2id_target
    for w in words:
      if w not in vocab:
        continue
      candidates2_w = self._candidate_generation_infer.candidates(w, source2target)
      candidates2.extend(candidates2_w)
      candidates1.extend([w] * len(candidates2_w))

    if source2target:
      examples = {'source/word': candidates1,
                  'target/word': candidates2}
    else:
      examples = {'source/word': candidates2,
                  'target/word': candidates1}
    if candidates1:
      input_feeds = self.input_feeds(examples, mode=Classifier.PREDICT)
      outputs = self.outputs(mode=Classifier.PREDICT)
      probs = self._sess.run(outputs, input_feeds)
    else:
      probs = []
    i = 0
    translations = []
    for w in words:
      if w not in vocab:
        translations.append([])
        continue
      translations_w = []
      while i < len(candidates1) and candidates1[i] == w:
        if np.isscalar(probs) and i == 0:
          prob = probs
        else:
          prob = probs[i]
        w_pred = candidates2[i]
        if prob > threshold:
          translations_w.append((w_pred, prob))
        i += 1

      translations.append(sorted(translations_w, key=operator.itemgetter(1), reverse=True))
    return translations

  @property
  def session(self):
    return self._sess

  def get_features(self, examples):
    return self.training_examples(examples)

  def get_oov_words(self, words, language='target'):
    if language == 'target':
      vocab = self._word2id_target
    else:
      vocab = self._word2id_source
    return [w for w in words if w not in vocab]


class Skipgram(object):
  def __init__(self, session, vocabulary, counts, num_samples, features_component, name=''):
    self._sess = session
    self._vocabulary = vocabulary
    self._counts = counts
    self._num_samples = num_samples
    self._features_component = features_component
    self._name = name
    features = features_component.output
    self._sg_loss = skipgram.SkipgramNCELoss(vocabulary, num_samples, counts, name)
    self._sg_loss_component = self._sg_loss(features)
    self._sg_loss = self._sg_loss_component.output

  @property
  def loss_variable(self):
    return self._sg_loss

  def get_features(self, examples):
    # examples 2 raw features
    # prefix + raw feature name (raw feature names: word)
    prefix = self._name + '/' if self._name else ''
    center_words, context_words = zip(*examples)
    batch = {
      prefix + 'center/word': center_words,
      prefix + 'context/word': context_words
    }
    return batch

  def input_feeds(self, features):
    feeds = self._features_component.input_feeds(**features)
    feeds.update(self._sg_loss_component.input_feeds(**features))
    return feeds


class MultitaskModel(object):
  def __init__(self, session, models, weights, optimizer=None):
    self._models = models
    self._losses = [m.loss_variable for m in models]
    losses = tf.stack(self._losses)
    weights = tf.constant(weights, dtype=tf.float32)
    loss = tf.reduce_sum(tf.multiply(losses, weights))
    self._loss = loss
    optimizer = tf.train.AdamOptimizer() if optimizer is None else optimizer
    gvs = optimizer.compute_gradients(self._loss)
    capped_gvs = [(tf.clip_by_norm(grad, 3), var) for grad, var in gvs]
    self._train_op = optimizer.apply_gradients(capped_gvs)
    self._session = session
    self._loss_value = 0.
    session.run(tf.initialize_all_variables())

  def step(self, examples_per_model):
    inputs = self.input_feeds(examples_per_model)
    outputs = [self._train_op, self._loss] + self._losses
    outputs = self._session.run(outputs, inputs)
    self._loss_value = outputs[1]
    self._loss_values = outputs[2:]
    for i, m in enumerate(self._models):
      m._loss_value = self._loss_values[i]

  def loss(self):
    return self._loss_value

  def losses(self):
    return self._loss_values

  def input_feeds(self, examples_per_model):
    input_feeds = {}
    for m, examples in zip(self._models, examples_per_model):
      input_features = m.get_features(examples)
      input_feeds.update(m.input_feeds(input_features))
    return input_feeds

  def num_tasks(self):
    return len(self._models)

  @property
  def session(self):
    return self._session
