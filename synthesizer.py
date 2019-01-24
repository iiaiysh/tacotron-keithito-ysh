import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence, text2list, add_tilde, list_pad_to_numpy
from util import audio
import time
import tensorflow as tf

class Synthesizer:
  def __init__(self,model_name='tacotron',reuse=None):
    print('Constructing Synthesizer model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')
    filenames = tf.placeholder(tf.string, [None], 'filenames')
    with tf.variable_scope('model',reuse=reuse) as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths, filenames)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])
      
      self.wav_outputs = tf.map_fn(audio.inv_spectrogram_tensorflow, self.model.linear_outputs)


  def load(self, checkpoint_path):
    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text):
    text = add_tilde(text)
    print('synthesize:',text)
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    wav = self.session.run(self.wav_output, feed_dict=feed_dict)
    wav = audio.inv_preemphasis(wav)
    wav = wav[:audio.find_endpoint(wav)]
    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue()

  def synthesize_fromlist(self, text):
    text_list = text2list(text)
    text_list = [ add_tilde(item) for item in text_list]
    print('synthesize from list:',text_list)

    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq_list = [text_to_sequence(text, cleaner_names) for text in text_list]
    seq_np = list_pad_to_numpy(seq_list)
    seq_len_list = [len(seq) for seq in seq_list]

    wav_list = []
    # t1 = time.time()
    seq = text_to_sequence(text, cleaner_names)
    feed_dict = {
      self.model.inputs: np.asarray(seq_np, dtype=np.int32),
      self.model.input_lengths: np.asarray(seq_len_list, dtype=np.int32)
    }
    wavs = self.session.run(self.wav_outputs, feed_dict=feed_dict)

    for wav in wavs:
      wav = audio.inv_preemphasis(wav)
      wav = wav[:audio.find_endpoint(wav)]
      wav_list.append(wav)

    # print('from_list_1',time.time()-t1)

    out = io.BytesIO()
    audio.save_wav_list(wav_list, out)
    return out.getvalue()


  def synthesize_fromlist_2(self, text):
    text_list = text2list(text)
    text_list = [ add_tilde(item) for item in text_list]
    print('synthesize from list:',text_list)
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    wav_list=[]

    # t1 = time.time()
    for text in text_list:
      seq = text_to_sequence(text, cleaner_names)
      feed_dict = {
        self.model.inputs: [np.asarray(seq, dtype=np.int32)],
        self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
      }
      wav= self.session.run(self.wav_output,feed_dict=feed_dict)

      wav = audio.inv_preemphasis(wav)
      wav = wav[:audio.find_endpoint(wav)]
      wav_list.append(wav)
    # print('from_list_2',time.time()-t1)

    out = io.BytesIO()
    audio.save_wav_list(wav_list, out)
    return out.getvalue()