
import utils
import string
import collections
import tensorflow as tf

class Test():
  def __init__(self, args):

  def load_data_test():
    data = utils.load_data("data.txt")
    assert data[0] == "qeubyajvrhwcdxgszmftk"
    assert data[1] == "ktfmzsgxdcwhrvjaybueq"


def load_data_test():
  data = utils.load_data("data.txt")
  assert data[0] == "qeubyajvrhwcdxgszmftk"
  assert data[1] == "ktfmzsgxdcwhrvjaybueq"
  return data

def build_vocab_test(data):
  index2word, word2index = utils.build_vocab_with_nltk(data)
  assert sorted(index2word) == ["</s>","<s>","<unk>"] + [ch for ch in string.ascii_lowercase]
  return index2word, word2index

def tokenize_data(data):
  tokens = []
  for line in data:
    tokens.append([ch for ch in line])
  return tokens

def tokenize_data_test(data):
  assert data[2] == ['e','l','w','j','r']

class Limits(
  collections.namedtuple("Limits",
    ("q_max_len", "a_max_len", "q_min_len", "a_min_len"))):
  pass

def get_data_limits(Limits):
  Limits.q_max_len, Limits.a_max_len, Limits.q_min_len, Limits.a_min_len = 26, 26, 1, 1

def split_data_test(data, Limits):
  queries, answers = split_data(data, Limits)
  return queries, answers
  
