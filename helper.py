
import tensorflow as tf
import numpy as np
import collections
import seq2seq
import utils
import time

class TrainModel(collections.namedtuple("TrainModel", ("graph", "model"))):
  pass

def build_train_model(args, name="train_model"):
  graph = tf.Graph()
  with graph.as_default():
    model = seq2seq.Model(args, mode=tf.contrib.learn.ModeKeys.TRAIN, name=name)
    return TrainModel(graph=graph, model=model)

class EvalModel(collections.namedtuple("EvalModel", ("graph", "model"))):
  pass

def build_eval_model(args, name="eval_model"):
  graph = tf.Graph()
  with graph.as_default():
    model = seq2seq.Model(args, mode=tf.contrib.learn.ModeKeys.EVAL, name=name)
    return EvalModel(graph=graph, model=model)

class InferModel(collections.namedtuple("InferModel", ("graph", "model"))):
  pass

def build_infer_model(args, name="infer_model"):
  graph = tf.Graph()
  with graph.as_default():
    model = seq2seq.Model(args, mode=tf.contrib.learn.ModeKeys.INFER, name=name)
    return InferModel(graph=graph, model=model)

def load_model(model, latest_ckpt, session, name):
  start_time = time.time()
  model.saver().restore(session, latest_ckpt)
  utils.print_out("loaded %s model parameters from %s, time %.2fs" %
      (name, ckpt, time.time() - start_time))
  return model

def create_or_load_model(model, ckpt_dir, session, name):
  latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
  if latest_ckpt:
    model = load_model(latest_ckpt)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    utils.print_out("created %s model with fresh parameters, time %.2fs" 
                        % (name, time.time() - start_time))

  global_step = session.run(model.global_step)
  return model, global_step