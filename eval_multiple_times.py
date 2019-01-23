import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
import tensorflow as tf
from util import audio
import numpy as np
sentences = [
"Welcome to the inception institute of artificial intelligence, my name is Tony Robbins! I am kidding, I am not really Tony Robbins. and I never said this. I am still a work in progress. so, don't mind any strange artifacts you might hear."
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return base_dir


def run_eval(args):
  #fmin_list=[125,115,105,95,85,75,65,55]
  #fmax_list=[7600,6600,5600,4600,3600]
  
  base_path = args.base_path
  
  synth = Synthesizer(reuse=tf.AUTO_REUSE)
 
  ckpt_list = os.listdir(base_path)
  print(ckpt_list)
  ckpt_list = [item[:-5] for item in ckpt_list if (item.startswith('model') and item.endswith('meta'))]
  print(ckpt_list)
  #print(hparams_debug_string())
  for ckpt in ckpt_list:
    step = ckpt.split('-')[1]
    checkpoint = f'{base_path}/{ckpt}'
    #print('load checkpoint...',checkpoint)
    synth.load(checkpoint)
    for i, text in enumerate(sentences):
      path = f'{base_path}-{step}-welcome.wav'
      print('Synthesizing: %s' % path)
      with open(path, 'wb') as f:
        f.write(synth.synthesize(text))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_path', required=True, help='Path to model checkpoint')
  parser.add_argument('--mel_targets')
  parser.add_argument('--reference_audio' )
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
