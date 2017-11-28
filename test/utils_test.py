
import utils
import string
import collections
import tensorflow as tf

class Limits(
  collections.namedtuple("Limits",
    ("q_max_len", "a_max_len", "q_min_len", "a_min_len"))):
  pass

def get_data_limits(Limits):
  Limits.q_max_len, Limits.a_max_len, Limits.q_min_len, Limits.a_min_len = 26, 26, 1, 1

class TestRawDataProcess():
  def __init__(self, data_dir):
    self.data_dir = data_dir

    data = self.load_data_test()
    index2word, _ = self.build_vocab_test(data)
    self.save_vocab_list('vocab.txt', index2word)
    token_data = self.tokenize_data(data)
    self.tokenize_data_test(token_data)
    queries, answers = self.split_data_test(token_data, Limits)
    
    utils.save_word_list("source.txt", queries)
    utils.save_word_list("target.txt", answers)
    
    print "Pass the test successfully"

  def load_data_test(self):
    data = utils.load_data(self.data_dir)
    assert data[0] == "ipoamtgyuwcrezvsdkhjb"
    assert data[1] == "abcdeghijkmoprstuvwyz"
    return data

  def build_vocab_test(self, data):
    index2word, word2index = utils.build_vocab_with_nltk(data)
    assert sorted(index2word) == ["</s>","<s>","<unk>"] + [ch for ch in string.ascii_lowercase]
    return index2word, word2index

  def save_vocab_list(self, file_dir, index2word):
    f = open(file_dir, "w")
    for word in index2word:
      f.write("".join(word) + "\n")
    f.close()

  def tokenize_data(self, data):
    tokens = []
    for line in data:
      tokens.append([ch for ch in line])
    return tokens

  def tokenize_data_test(self, data):
    assert data[6] == ['b', 'z', 'd', 'l', 'x', 'a', 'i', 'h', 't', 'y']

  def split_data_test(self, data, Limits):
    queries, answers = utils.split_data(data, Limits)
    assert queries[3] == ['b', 'z', 'd', 'l', 'x', 'a', 'i', 'h', 't', 'y']
    assert answers[3] == list(sorted(['b', 'z', 'd', 'l', 'x', 'a', 'i', 'h', 't', 'y']))
    return queries, answers

from tensorflow.python.ops import lookup_ops

class TestDatasetAPI():
  def __init__(self, src_data_dir, tgt_data_dir):
    self.src_data_dir = src_data_dir
    self.tgt_data_dir = tgt_data_dir
    self.batch_size = 3
    
    self.testGetIterator()

  def testGetIterator(self):
    tgt_vocab_table = src_vocab_table = lookup_ops.index_table_from_file(
      'vocab.txt', default_value=0)
    # src_data = utils.load_data(self.src_data_dir)
    # tgt_data = utils.load_data(self.tgt_data_dir)
    # src_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(src_data))
    # tgt_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(tgt_data))
    src_dataset = tf.data.TextLineDataset(self.src_data_dir)
    tgt_dataset = tf.data.TextLineDataset(self.tgt_data_dir)

    iterator = utils.get_iterator(
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        batch_size=self.batch_size,
        random_seed=123,
        shuffle=False,
        source_reverse=False)

    tables_initializer = tf.tables_initializer()
    source = iterator.source
    target_input = iterator.target_input
    target_output = iterator.target_output
    src_seq_len = iterator.source_sequence_length
    tgt_seq_len = iterator.target_sequence_length

    with tf.Session() as sess:
      sess.run(tables_initializer)
      sess.run(iterator.initializer)

      (source_v, src_len_v, target_input_v, target_output_v, tgt_len_v) = (
          sess.run((source, src_seq_len, target_input, target_output,
                    tgt_seq_len)))

      print source_v
      print src_len_v
      print target_input_v
      print target_output_v
      print tgt_len_v

if __name__ == '__main__':
  get_data_limits(Limits)
  #TestRawDataProcess("data_sort.txt")
  TestDatasetAPI("source.txt", "target.txt")
