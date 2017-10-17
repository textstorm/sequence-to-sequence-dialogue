
import argparse

def get_args():
  """
    The argument parser
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--random_seed', type=int, default=1013, help='Random seed')

  #data
  parser.add_argument('--data_dir', type=str, default='/root/textstorm/Seq2Seq/data/Twitter/chat.txt', help='Tdata directory')
  parser.add_argument('--log_dir', type=str, default='logs', help='Tensorboard logs directory')
  parser.add_argument('--save_dir', type=str, default='/root/textstorm/Seq2Seq/saves/', help='Train batch size')
  parser.add_argument('--whitelist', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz\' ')
  parser.add_argument('--max_words', type=int, default=10000)
  parser.add_argument('--q_max_len', type=int, default=20)
  parser.add_argument('--a_max_len', type=int, default=20)
  parser.add_argument('--q_min_len', type=int, default=1)
  parser.add_argument('--a_min_len', type=int, default=1)
  parser.add_argument('--batch_size', type=int, default=32, help='Train batch size')

  #model details
  parser.add_argument('--embedding_size', type=int, default=100, help='The dimension of embedding')
  parser.add_argument('--hidden_size', type=int, default=128, help='The dimension of rnn hidden layer')
  parser.add_argument('--num_layers', type=int, default=1, help='The layers number of rnn')
  parser.add_argument('--forget_bias', type=int, default=1, help='The residual layers number')
  parser.add_argument('--num_residual_layers', type=int, default=1, help='The residual layers number')
  parser.add_argument('--encoder_vocab_size', type=int, default=10003, help='The residual layers number')
  parser.add_argument('--encoder_embed_size', type=int, default=500, help='The residual layers number')
  parser.add_argument('--decoder_vocab_size', type=int, default=10003, help='The residual layers number')
  parser.add_argument('--decoder_embed_size', type=int, default=500, help='The residual layers number')
  parser.add_argument('--encoder_type', type=str, default="uni", help='The residual layers number')
  parser.add_argument('--beam_width', type=int, default=1, help='The residual layers number')

  #optimizer details
  parser.add_argument('--nb_epoch', type=int, default=1, help='The number of epoch')
  parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
  parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
  parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Max norm of gradient')
  parser.add_argument('--max_batch', type=int, default=512, help='Maximum examples of batch')
  parser.add_argument('--anneal_period', type=int, default=25, help='anneal period')
  parser.add_argument('--print_period', type=int, default=1, help='Print information period')
  parser.add_argument('--log_period', type=int, default=100, help='Log information period')

  return parser.parse_args()

