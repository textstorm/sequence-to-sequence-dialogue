
from collections import Counter
import numpy as np
import sys
import time

#data utils
def load_data(file_dir):
  print 'Read sentences from file'
  f = open(file_dir, 'r')
  sentences = []
  while True:
    sentence = f.readline()
    if not sentence:
      break

    sentence = sentence.strip().lower()
    sentences.append(sentence)
  f.close()
  return sentences

def filter_sentences(sentence, whitelist):
  """
    filter out the emoji in a sentence
    whitelist: 
  """
  return "".join([ch for ch in sentence if ch in whitelist])

def build_vocab(sentences, max_words=None):
  print "Buildding vocabulary..."
  word_count = Counter()
  for sentence in sentences:
    for word in sentence.split(" "):
      word_count[word] += 1

  print "The dataset has %d different words totally" % len(word_count)
  if not max_words:
    max_words = len(word_count)
  else:
    filter_out_words = len(word_count) - max_words

  word_dict = word_count.most_common(max_words)
  index2word = ["<unk>"] + ["<s>"] + ["</s>"] + [word[0] for word in word_dict]
  word2index = dict([(word, idx) for idx, word in enumerate(index2word)])

  print "%d words filtered out of the vocabulary and %d words in the vocabulary" % (filter_out_words, max_words)
  return index2word, word2index

def split_data(sentences, limits):
  queries = []
  answers = []
  num_example = len(sentences) // 2
  print "The dataset has %d queries and answers tuple" % num_example

  for i in range(0, len(sentences), 2):
    qlen, alen = len(sentences[i].split(" ")), len(sentences[i+1].split(" "))
    if qlen >= limits.q_min_len and alen >= limits.a_min_len:
      if qlen <= limits.q_max_len and alen <= limits.a_max_len:
        queries.append(sentences[i])
        answers.append(sentences[i+1])

  filtered_data_len = len(queries)
  filter_out_len = num_example - filtered_data_len
  print '%d tuple filtered out of raw data filter_out_len' % filter_out_len
  return queries, answers

def vectorize(queries,  answers, word2index, sort_by_len=False, verbose=True):
  """
    note: the dict is only 50K,words not in dict is 0
    queries: questions after vectorize
    answers: answers after vectorize
    if sort_by_len equal True, documents sorted by length 
  """
  vec_queries = []
  vec_answers = []
  for idx, (q, a) in enumerate(zip(queries, answers)):
    q_words = q.split(' ')
    a_words = a.split(' ')
    seq_q = [word2index[w] if w in word2index else 0 for w in q_words]
    seq_a = [word2index[w] if w in word2index else 0 for w in a_words]
    vec_queries.append(seq_q)
    vec_answers.append(seq_a)

    if verbose and (idx % 50000 == 0):
      print("Vectorization: processed {}".format(idx))

  def len_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))
  
  if sort_by_len:
    sort_index = len_argsort(vec_queries)
    vec_queries = [vec_queries[i] for i in sort_index]
    vec_answers = [vec_answers[i] for i in sort_index]

  return vec_queries, vec_answers

def padding_data(sentences):
  """
    in general,when padding data,first generate all-zero matrix,then for every
    sentence,0 to len(seq) assigned by seq,like pdata[idx, :lengths[idx]] = seq

      pdata: data after zero padding
      lengths: length of sentences
  """
  lengths = [len(s) for s in sentences]
  n_samples = len(sentences)
  max_len = np.max(lengths)
  pdata = np.zeros((n_samples, max_len)).astype('int32')
  for idx, seq in enumerate(sentences):
    pdata[idx, :lengths[idx]] = seq
  return pdata, lengths 

def get_batchidx(n_data, batch_size, shuffle=False):
  """
    batch all data index into a list
  """
  idx_list = np.arange(0, n_data, batch_size)
  if shuffle:
    np.random.shuffle(idx_list)
  bat_index = []
  for idx in idx_list:
    bat_index.append(np.arange(idx, min(idx + batch_size, n_data)))
  return bat_index

def get_batches(queries, answers, batch_size):
  """
    read all data into ram once
  """
  minibatches = get_batchidx(len(queries), batch_size)
  all_bat = []
  for minibatch in minibatches:
    q_bat = [queries[t] for t in minibatch]
    a_bat = [answers[t] for t in minibatch]
    q_pad, q_len = padding_data(q_bat)
    a_pad, a_len = padding_data(a_bat)
    all_bat.append((q_pad, q_len, a_pad, a_len))
  return all_bat

def print_time(s, start_time):
  """Take a start time, print elapsed duration, and return a new time."""
  print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
  sys.stdout.flush()
  return time.time()

def print_out(s, f=None, new_line=True):
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print(out_s, end="", file=sys.stdout)

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()

#tensorflow utils
def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto
