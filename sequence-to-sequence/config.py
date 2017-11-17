
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
  parser.add_argument('--whitelist', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz\' ', help="characters not filtered out")
  parser.add_argument('--q_max_len', type=int, default=20, help='The max first utterance length')
  parser.add_argument('--a_max_len', type=int, default=20, help='The max second utterance length')
  parser.add_argument('--q_min_len', type=int, default=1, help='The min first utterance length')
  parser.add_argument('--a_min_len', type=int, default=1, help='The min second utterance length')
  parser.add_argument('--batch_size', type=int, default=32, help='Train batch size')

  #model details
  parser.add_argument('--hidden_size', type=int, default=128, help='The dimension of rnn hidden layer')
  parser.add_argument('--num_layers', type=int, default=1, help='The layers number of rnn')
  parser.add_argument('--forget_bias', type=int, default=1, help='The forget bias')
  parser.add_argument('--encoder_vocab_size', type=int, default=10003, help='The vocab size of encoder')
  parser.add_argument('--encoder_embed_size', type=int, default=500, help='The embedding size of encoder')
  parser.add_argument('--decoder_vocab_size', type=int, default=10003, help='The vocab size of decoder')
  parser.add_argument('--decoder_embed_size', type=int, default=500, help='The embedding size of decoder')
  parser.add_argument('--encoder_type', type=str, default="uni", help='uni or bi direction')
  parser.add_argument('--beam_width', type=int, default=1, help='The beam width when infer')

  #optimizer details
  parser.add_argument('--nb_epoch', type=int, default=1, help='The number of epoch')
  parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
  parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
  parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Max norm of gradient')
  parser.add_argument('--anneal_period', type=int, default=25, help='anneal period')
  parser.add_argument('--print_period', type=int, default=1, help='Print information period')
  parser.add_argument('--log_period', type=int, default=100, help='Log information period')

  return parser.parse_args()
