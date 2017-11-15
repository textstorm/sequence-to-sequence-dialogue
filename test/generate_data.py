
import string
import random

data_base = string.ascii_lowercase
data_base = [ch for ch in data_base]

def make_data():
  data = []
  reverse_data = []
  sorted_data = []
  for i in range(10000):
    seq_len = random.randint(1,26)
    seq_idx = random.sample(range(len(data_base)), seq_len)
    seq = [data_base[j] for j in seq_idx]
    data.append("".join(seq))
    reverse_data.append("".join(reversed(seq)))
    sorted_data.append("".join(sorted(seq)))
  return data, reverse_data, sorted_data

def save_data(data_dir, data):
  f = open(data_dir, 'w')
  for src, tgt in data:
    f.write(src + '\n')
    f.write(tgt + '\n')
  f.close()

if __name__ == "__main__":
  data, reverse_data, sorted_data = make_data()
  save_data("data_reverse.txt", zip(data, reverse_data))
  save_data("data_sort.txt", zip(data, sorted_data))