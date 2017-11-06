
import collections
import utils

class Limits(
  collections.namedtuple("Limits", ("q_max_len", "a_max_len", "q_min_len", "a_min_len"))):
  pass
Limits.q_max_len, Limits.a_max_len, Limits.q_min_len, Limits.a_min_len = 30, 30, 1,1

def make_src_tgt_data():
  train_dir = "train.txt"
  eval_dir = "eval.txt"
  test_dir = "test.txt"

  train_data = utils.load_data(train_dir)
  eval_data = utils.load_data(eval_dir)
  test_data = utils.load_data(test_dir)

  train_data = utils.filter_sentences_with_punct(train_data)
  eval_data = utils.filter_sentences_with_punct(eval_data)
  test_data = utils.filter_sentences_with_punct(test_data)

  token_train_data = utils.tokenize_data(train_data)
  token_eval_data = utils.tokenize_data(eval_data)
  token_test_data = utils.tokenize_data(test_data)

  index2word, word2index = utils.build_vocab_with_nltk(token_train_data, 10000)

  train_src_data, train_tgt_data = utils.split_data(token_train_data, Limits)
  eval_src_data, eval_tgt_data = utils.split_data(token_eval_data, Limits)
  test_src_data, test_tgt_data = utils.split_data(token_test_data, Limits)

  utils.save_data("train_src_data.txt", train_src_data)
  utils.save_data("train_tgt_data.txt", train_tgt_data)
  utils.save_data("eval_src_data.txt", eval_src_data)
  utils.save_data("eval_tgt_data.txt", eval_tgt_data)
  utils.save_data("test_src_data.txt", test_src_data)
  utils.save_data("test_tgt_data.txt", test_tgt_data)
  utils.save_data("vocab.txt", index2word)

if __name__ == "__main__":
  make_src_tgt_data()