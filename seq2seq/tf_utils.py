
import collections
import tensorflow as tf

class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "target_output"))):
  pass


class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"))):
def get_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 source_reverse,
                 num_threads=4):

  if not output_buffer_size:
    output_buffer_size = batch_size * 1000

  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant("<s>")), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant("</s>")), tf.int32)

  src_tgt_dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))

  #src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed)

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (tf.string_split([src]).value, tf.string_split([tgt]).values),
    num_threads=num_threads)

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                      tf.cast(src_vocab_table.lookup(tgt), tf.int32)),
    num_threads=num_threads, output_buffer_size=output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (src,
                      tf.concat(([tgt_sos_id, tgt]), 0),
                      tf.concat(([tgt, tgt_eos_id]), 0)),
    num_threads=num_threads, output_buffer_size=output_buffer_size)

  batch_iterator = src_tgt_dataset.make_initializable_iterator()
  source, target_input, target_output = batch_iterator.get_next()
  return BatchedInput(
    initializer=batch_iterator.initializer,
    source=source,
    )
  def batching_fun(x):
    return x.padded_batch(
      batch_size,
      padded_shapes=)