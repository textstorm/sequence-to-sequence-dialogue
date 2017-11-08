
import collections
import tensorflow as tf

class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"))):
  pass

def get_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 random_seed,
                 source_reverse=False,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 num_shards=1,
                 shard_index=0):

  if not output_buffer_size:
    output_buffer_size = batch_size * 1000

  unk_id = tf.cast(tgt_vocab_table.lookup(tf.constant("<unk>")), tf.int32)
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant("<s>")), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant("</s>")), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)

  #src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed)

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values),
    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  if source_reverse:
    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.reverse(src, axis=[0]), tgt),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                      tf.cast(src_vocab_table.lookup(tgt), tf.int32)),
    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (src,
                      tf.concat(([tgt_sos_id], tgt), 0),
                      tf.concat((tgt, [tgt_eos_id]), 0)),
    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt_in, tgt_out: (src,
                                  tgt_in,
                                  tgt_out,
                                  tf.size(src),
                                  tf.size(tgt_in)),
    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len

        padding_values=(
            unk_id,  # src
            unk_id,  # tgt_input
            unk_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused

  batch_dataset = batching_func(src_tgt_dataset)
  batch_iterator = batch_dataset.make_initializable_iterator()
  (source, target_input, target_output, src_seq_len, tgt_seq_len) = (batch_iterator.get_next())
  return BatchedInput(
    initializer=batch_iterator.initializer,
    source=source,
    target_input=target_input,
    target_output=target_output,
    source_sequence_length=src_seq_len,
    target_sequence_length=tgt_seq_len)
