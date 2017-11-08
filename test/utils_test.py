
import utils
import string
import collections
import tensorflow as tf

class Limits(
  collections.namedtuple("Limits",
    ("q_max_len", "a_max_len", "q_min_len", "a_min_len"))):
  pass

class TestRawDataProcess():
  def __init__(self, data_dir):
    self.data_dir = data_dir

    data = load_data_test()
    index2word, _ = build_vocab_test(data)
    token_data = token_data(data)
    tokenize_data_test(token_data)
    get_data_limits(Limits)

    utils.save_word_list("source.txt", queries)
    utils.save_word_list("target.txt", answers)

  def load_data_test(self):
    data = utils.load_data(self.data_dir)
    assert data[0] == "qeubyajvrhwcdxgszmftk"
    assert data[1] == "ktfmzsgxdcwhrvjaybueq"
    return data

  def build_vocab_test(self, data):
    index2word, word2index = utils.build_vocab_with_nltk(data)
    assert sorted(index2word) == ["</s>","<s>","<unk>"] + [ch for ch in string.ascii_lowercase]
    return index2word, word2index

  def tokenize_data(self, data):
    tokens = []
    for line in data:
      tokens.append([ch for ch in line])
    return tokens

  def tokenize_data_test(self, data):
    assert data[2] == ['e','l','w','j','r']

  def get_data_limits(self, Limits):
    Limits.q_max_len, Limits.a_max_len, Limits.q_min_len, Limits.a_min_len = 26, 26, 1, 1

  def split_data_test(self, data, Limits):
    queries, answers = utils.split_data(data, Limits)
    assert queries[1] == ['e','l','w','j','r']
    assert answers[1] == list(reversed(['e','l','w','j','r']))
    return queries, answers

if __name__ == '__main__':
  Limits()
  TestRawDataProcess()
