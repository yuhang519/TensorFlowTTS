import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

def main():
  # initialize fastspeech model.
  fs_config = AutoConfig.from_pretrained('./examples/fastspeech/conf/fastspeech.v1.yaml')
  fastspeech = TFAutoModel.from_pretrained(
      config=fs_config,
      pretrained_path="./examples/fastspeech/pretrained/model-195000.h5"
  )


  # initialize melgan model
  melgan_config = AutoConfig.from_pretrained('./examples/melgan/conf/melgan.v1.yaml')
  melgan = TFAutoModel.from_pretrained(
      config=melgan_config,
      pretrained_path="./examples/melgan/checkpoints/generator-1500000.h5"
  )


  # inference
  processor = AutoProcessor.from_pretrained(pretrained_path="./test/files/ljspeech_mapper.json")

  # text to be converted
  ids = processor.text_to_sequence("Hi boss, the current version of our TTS has been able to convert text into speech. I am still working on APIs for easier usage.")
  ids = tf.expand_dims(ids, 0)
  
  # fastspeech inference
  masked_mel_before, masked_mel_after, duration_outputs = fastspeech.inference(
    ids,
    speaker_ids=tf.zeros(shape=[tf.shape(ids)[0]], dtype=tf.int32),
    speed_ratios=tf.constant([1.0], dtype=tf.float32)
  )

  # melgan inference
  #audio_before = melgan.inference(masked_mel_before)[0, :, 0]
  audio_after = melgan.inference(masked_mel_after)[0, :, 0]

  # save to file
  #sf.write('./audio_before.wav', audio_before, 22050, "PCM_16")
  sf.write('./audio_after.wav', audio_after, 22050, "PCM_16")

if __name__=='__main__':
    main()
