import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


sentences = [
  # # From July 8, 2017 New York Times:
  # 'Scientists at the CERN laboratory say they have discovered a new particle.',
  # 'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
  # 'President Trump met with other leaders at the Group of 20 conference.',
  # 'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
  # # From Google's Tacotron example page:
  # 'Generative adversarial network or variational auto-encoder.',
  # 'The buses aren\'t the problem, they actually provide a solution.',
  # 'Does the quick brown fox jump over the lazy dog?',
  # 'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
  # "Tony Robbins is an entrepreneur,",
  "Tony Robbins is an entrepreneur, best-selling author, philanthropist and the nation’s #1 Life and Business Strategist. A recognized authority on the psychology of leadership, negotiations and organizational turnaround, he has served as an advisor to leaders around the world for more than 40 years."
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    path = '%s-%d.wav' % (base_path, i)
    path2 = '%s-%d-2.wav' % (base_path, i)

    print('Synthesizing: %s' % path)

    with open(path2, 'wb') as f:    
      f.write(synth.synthesize_fromlist_2(text))
    with open(path, 'wb') as f:
      f.write(synth.synthesize_fromlist(text))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
