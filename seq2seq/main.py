
import tensorflow as tf
import numpy as np
import collections
import utils
import config
import seq2seq
import helper
import time
import os

class Limits(
  collections.namedtuple("Limits", ("q_max_len", "a_max_len", "q_min_len", "a_min_len"))):
  pass

def prepare_quries_answers(args):
  chat_data = utils.load_data(args.data_dir)
  chat_data = utils.filter_sentences(chat_data, args.whitelist)
  index2word, word2index = utils.build_vocab(chat_data, max_words=args.max_words)
  Limits.q_max_len, Limits.a_max_len, Limits.q_min_len, Limits.a_min_len = args.q_max_len, \
      args.a_max_len, args.q_min_len, args.a_min_len
  queries, answers = utils.split_data(chat_data, Limits)
  queries, answers = utils.vectorize(queries,  answers, word2index, sort_by_len=True)

  return queries, answers, index2word, word2index

def infer_test(infer_model, infer_sess, args, queries, index2word):
  with infer_model.graph.as_default():
    new_infer_model, global_step = helper.create_or_load_model(
                  infer_model.model, args.save_dir, infer_sess, name="infer")
    decode_id = np.random.randint(0, len(queries) - 1)
    encoder_input = queries[decode_id].reshape(1, -1)
    encoder_input_len = [len(encoder_input)]
    sample_id = new_infer_model.infer(encoder_input, encoder_input_len, infer_sess)
    utils.print_out(utils.de_vectorize(encoder_input, index2word))
    utils.print_out(utils.de_vectorize(sample_id, index2word))

def main(args):

  utils.print_out("Sequence-to-Sequence dialogue")
  utils.print_out("- " * 50)
  queries, answers, index2word, word2index = prepare_quries_answers(args)
  batch_data = utils.get_batches(queries, answers, args.batch_size)

  config_proto = utils.get_config_proto()

  with tf.device("/gpu:0"):
    if not os.path.exists("../saves"):
      os.mkdir("../saves")
    save_dir = args.save_dir

    train_model = helper.build_train_model(args)
    eval_model = helper.build_eval_model(args)
    infer_model = helper.build_infer_model(args)

    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)
    infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)

    with train_model.graph.as_default():
      new_train_model, global_step = helper.create_or_load_model(
                  train_model.model, args.save_dir, train_sess, name="train")

    for epoch in range(1, args.nb_epoch + 1):
      utils.print_out("Epoch: %d start" % epoch)
      utils.print_out("- " * 50)

      loss = 0.0
      start_train_time = time.time()
      for idx, batch in enumerate(batch_data):
        queries, qry_lens, answers, ans_lens = batch
        loss_t = new_train_model.train(queries, qry_lens, answers, ans_lens, train_sess)

        if idx % 100 == 0:
          print "Epoch: ", '%01d' % epoch, "Batch: ", '%04d' % idx, "Loss: ", '%9.9f' % loss_t

      #saver.save(session, save_dir, global_step=new_train_model.global_step)

    infer_test(infer_model, infer_sess, args, queries, index2word)

if __name__ == "__main__":
  args = config.get_args()
  main(args)