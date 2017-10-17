
from tensorflow.python.layers import core as layers_core
import tensorflow as tf
import numpy as np
import collections
import time

class Model(object):
  def __init__(self, args, mode, name="Model"):
    self._num_layers = args.num_layers
    self._encoder_vocab_size = args.encoder_vocab_size
    self._encoder_embed_size = args.encoder_embed_size
    self._decoder_vocab_size = args.decoder_vocab_size
    self._decoder_embed_size = args.decoder_embed_size
    self._hidden_size = args.hidden_size
    self._forget_bias = args.forget_bias
    self._num_layers = args.num_layers
    self._dropout = args.dropout
    self._encoder_type = args.encoder_type #bi-lstm or not
    self._beam_width = args.beam_width
    self._max_grad_norm = args.max_grad_norm

    self.batch_size = args.batch_size
    self.mode = mode

    self._optimizer = tf.train.AdamOptimizer()
    self.global_step = tf.get_variable('global_step', [], 'int32', 
                              tf.constant_initializer(0), trainable=False)

    with tf.variable_scope("seq2seq"):
      with tf.variable_scope("decoder"):
        self.output_layer = layers_core.Dense(
            self._decoder_vocab_size, use_bias=False, name="output_projection")

    self._build_placeholder()
    self._build_forward()

    self.saver = tf.train.Saver(tf.global_variables())

  def _build_placeholder(self):
    with tf.name_scope("data"):
      self.encoder_input = tf.placeholder(tf.int32, [None, None], name="encoder_data")
      self.encoder_length = tf.placeholder(tf.int32, [None], name="encoder_length")
      self.decoder_input = tf.placeholder(tf.int32, [None, None], name="decoder_data")
      self.decoder_length = tf.placeholder(tf.int32, [None], name="decoder_length")

    with tf.name_scope("hpara"):
      self.lr = tf.placeholder(tf.float32, [], name="learning_rate")

  def _build_forward(self):
    with tf.variable_scope("seq2seq"):
      encoder_output, encoder_state = self._build_encoder()
      self.logits, self.sample_id, self.final_state = self._build_decoder(encoder_state)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        self._build_loss()
        self._build_train()
      else:
        self.loss = None

  def _build_encoder(self):
    with tf.variable_scope("encoder") as scope:
      encoder_embed = self._build_embedding(self._encoder_vocab_size, self._encoder_embed_size, "en_embed")
      encoder_embed_inp = tf.nn.embedding_lookup(encoder_embed, self.encoder_input)

      if self._encoder_type == "uni":
        encoder_cell = self._build_encoder_cell(self._hidden_size, self._forget_bias, 
                                                self._num_layers, self.mode, self._dropout)
        encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_embed_inp, 
                                              dtype=tf.float32, sequence_length=self.encoder_length)

      elif self._encoder_type == "bi":
        num_bi_layers = self._num_layers / 2
        fw_cell = self._build_encoder_cell(self._hidden_size, self._forget_bias, 
                                            num_bi_layers, self.mode, self._dropout)
        bw_cell = self._build_encoder_cell(self._hidden_size, self._forget_bias,
                                            num_bi_layers, self.mode, self._dropout)
        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, encoder_embed_inp,
                                              dtype=tf.float32, sequence_length=self.encoder_length)
        bi_outputs = tf.concat(bi_outputs, -1)

        if num_bi_layers == 1:
          encoder_state = bi_state
        else:
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_state[0][layer_id])   #forward
            encoder_state.append(bi_state[1][layer_id])   #backward
          encoder_state = tuple(encoder_state)

      else:
        raise ValueError("Unknown encoder_type %s" % self._encoder_type)

    return encoder_output, encoder_state

  def _build_decoder(self, encoder_state):
    with tf.variable_scope("decoder") as scope:
      decoder_embed = self._build_embedding(self._decoder_vocab_size, self._decoder_embed_size, "de_embed")
      decoder_embed_inp = tf.nn.embedding_lookup(decoder_embed, self.decoder_input)

      decoder_cell, decoder_initial_state = self._build_decoder_cell(self._hidden_size, 
                                  self._forget_bias, self._num_layers, self.mode, encoder_state, self._dropout)

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_embed_inp, self.decoder_length, name="de_helper")
        my_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state)
        output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder)

        sample_id = output.sample_id
        logits = self.output_layer(output.rnn_output)

      else:
        #unk:0 sos:1 eos:2
        start_tokens = tf.fill([self.batch_size], 1)
        end_token = 2

        if self._beam_width > 0:
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=decoder_embed, start_tokens=start_tokens, 
              end_token=end_token, initial_state=decoder_initial_state, output_layer=self.output_layer, beam_width=self._beam_width)

        else:
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embed, start_tokens, end_token)
          my_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=self.output_layer)

        output, final_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder)

        if self._beam_width > 0:
          logits = tf.no_op()
          sample_id = output.predicted_ids
        else:
          logits = output.rnn_output
          sample_id = output.sample_id

    return logits, sample_id, final_state

  def _build_encoder_cell(self, num_units, forget_bias, num_layers, mode, dropout):

    return self._build_rnn_cell(num_units, forget_bias, num_layers, mode, dropout)

  def _build_decoder_cell(self, num_units, forget_bias, num_layers, mode, encoder_state, dropout):
    cell = self._build_rnn_cell(num_units, forget_bias, num_layers, mode, dropout)

    if self._beam_width > 0:
      encoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, self._beam_width)
    else:
      encoder_initial_state = encoder_state
    return cell, encoder_initial_state

  def _build_rnn_cell(self, num_units, forget_bias, num_layers, mode, dropout):
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, forget_bias=forget_bias)
    if dropout > 0.0:
      cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - dropout))

    cell_list = []
    for i in range(num_layers):
      cell_list.append(cell)

    if num_layers == 1:
      return cell_list[0]
    else:
      return tf.contrib.rnn.MultiRNNCell(cell_list)

  def get_max_time(self, tensor):
    time_axis = 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  def _build_loss(self):
    with tf.name_scope("train"):
      max_len = self.get_max_time(self.decoder_input)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_input, logits=self.logits)
      weight = tf.sequence_mask(self.decoder_length, max_len, dtype=self.logits.dtype)

      self.loss_op = tf.reduce_sum(loss * weight) / tf.to_float(self.batch_size)

  def _build_train(self):
    with tf.name_scope("train"):
      grads_and_vars = self._optimizer.compute_gradients(self.loss_op)
      grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self._optimizer.apply_gradients(grads_and_vars, global_step=self.global_step, name="train_op")

  def train(self, encoder_input, encoder_length, decoder_input, decoder_length, session):
    feed_dict = {self.encoder_input: encoder_input, self.encoder_length: encoder_length,
        self.decoder_input: decoder_input, self.decoder_length: decoder_length}

    loss, _ = session.run([self.loss_op, self.train_op], feed_dict=feed_dict)
    return loss

  def infer(self, encoder_input, encoder_length, session):
    feed_dict = {self.encoder_input: encoder_input, self.encoder_length: encoder_length}
    logits, sample_id = session.run([self.logits, self.sample_id], feed_dict=feed_dict)
    return sample_id

  def _weight_variable(self, shape, name, initializer=None):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

  def _build_embedding(self, vocab_size, embed_size, name):
    with tf.variable_scope("embedding") as scope:
      embedding = self._weight_variable([vocab_size, embed_size], name=name)
    return embedding

  def _build_embedding_for_encoder_and_decoder(self, en_vocab_size, de_vocab_size, 
                                              en_embed_size, de_embed_size, scope):
    with tf.variable_scope(scope or "embeddings") as scope:
      encoder_embed = self._weight_variable([en_vocab_size, en_embed_size], name="embedding_encoder")
      decoder_embed = self._weight_variable([de_vocab_size, de_embed_size], name="embedding_decoder")
    return encoder_embed, decoder_embed

  def _bias_variable(self, shape, name, initializer=None):
    initializer = tf.constant_initializer(0.)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)
