
import tensorflow as tf
import collections
import utils
import config
import seq2seq
import time
import os

class Limits(
  collections.namedtuple("Limits", ("q_max_len", "a_max_len", "q_min_len", "a_min_len"))):
  pass

def main(args):

  tf.logging.info("Sequence-to-Sequence dialogue")
  tf.logging.info('Loading data files.')
  chat_data = utils.load_data(args.data_dir)
  print "filtering emoji out of raw data..."
  chat_data = [utils.filter_sentences(sentence, args.whitelist) for sentence in chat_data]
  index2word, word2index = utils.build_vocab(chat_data, max_words=args.max_words)
  Limits.q_max_len, Limits.a_max_len, Limits.q_min_len, Limits.a_min_len = args.q_max_len, \
      args.a_max_len, args.q_min_len, args.a_min_len
  queries, answers = utils.split_data(chat_data, Limits)
  queries, answers = utils.vectorize(queries,  answers, word2index, sort_by_len=True)

  batch_data = utils.get_batches(queries, answers, args.batch_size)

  config_proto = utils.get_config_proto()

  with tf.device("/gpu:0"):
    if not os.path.exists("../saves"):
      os.mkdir("../saves")
    save_dir = args.save_dir

    train_model = seq2seq.build_train_model(args)
    eval_model = seq2seq.build_eval_modeli(args)
    infer_model = seq2seq.build_infer_model(args)

    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)
    infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)

    with train_model.graph.as_default():
      new_train_model, global_step = seq2seq.create_or_load_model(
                  train_model.model, args.save_dir, train_sess, name="train")

    for epoch in range(1, args.nb_epoch + 1):
      print "Epoch: %d start" % epoch
      print "- " * 50

      loss = 0.0
      start_train_time = time.time()
      for idx, batch in enumerate(batch_data):
        queries, qry_lens, answers, ans_lens = batch
        loss_t = new_train_model.train(queries, qry_lens, answers, ans_lens, train_sess)

        if idx % 100 == 0:
          print "Epoch: ", '%01d' % epoch, "Batch: ", '%04d' % idx, "Loss: ", '%9.9f' % loss_t

      saver.save(session, save_dir, global_step=new_train_model.global_step)

if __name__ == "__main__":
  args = config.get_args()
  main(args)