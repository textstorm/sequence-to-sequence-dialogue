
import utils
import tensorflow as tf

class Test():
  def __init__(self, args):

  def load_data_test():
    data = utils.laod_data("data.txt")
    assert data[0] == "qeubyajvrhwcdxgszmftk"
    assert data[1] == "ktfmzsgxdcwhrvjaybueq"


def load_data_test():
  data = utils.laod_data("data.txt")
  assert data[0] == "qeubyajvrhwcdxgszmftk"
  assert data[1] == "ktfmzsgxdcwhrvjaybueq"

def build_vocab_test():