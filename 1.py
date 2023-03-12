from pathlib import Path

import soundfile as sf

import os

from paddlespeech.t2s.exps.syn_utils import get_am_output

from paddlespeech.t2s.exps.syn_utils import get_frontend

from paddlespeech.t2s.exps.syn_utils import get_predictor

from paddlespeech.t2s.exps.syn_utils import get_voc_output



def get_text_dict(name:str,txtname:str):

  ff = open(txtname,"r",encoding="utf-8")

  msg = ff.read()

  ff.close()

  text_list = msg.split("\n")

  text_dict = {}

  num = 0

  for i in text_list:

    text_dict[name+str(num)] = i

    num+=1

    print(f"{name}text:{num}")

  return text_dict



def the_main(text_dict):

  # frontend

  frontend = get_frontend(

    lang="mix",

    phones_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),

    tones_dict=None

  )



  # am_predictor

  am_predictor = get_predictor(

    model_dir=am_inference_dir,

    model_file="fastspeech2_mix" + ".pdmodel",

    params_file="fastspeech2_mix" + ".pdiparams",

    device=device)



  # voc_predictor

  voc_predictor = get_predictor(

    model_dir=voc_inference_dir,

    model_file="pwgan_aishell3" + ".pdmodel",  # 这里以 pwgan_aishell3 为例子，其它模型记得修改此处模型名称

    params_file="pwgan_aishell3" + ".pdiparams",

    device=device)



  output_dir = Path(wav_output_dir)

  output_dir.mkdir(parents=True, exist_ok=True)



  sentences = list(text_dict.items())



  merge_sentences = True

  fs = 24000

  for utt_id, sentence in sentences:

    am_output_data = get_am_output(

      input=sentence,

      am_predictor=am_predictor,

      am="fastspeech2_mix",

      frontend=frontend,

      lang="mix",

      merge_sentences=merge_sentences,

      speaker_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),

      spk_id=0, )



    # 保存文件

    sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=fs)

  return



if __name__ == '__main__':

  #模型路径

  am_inference_dir = "man1"

  #声码器路径，这里以 pwgan_aishell3 为例子

  voc_inference_dir = "pwgan_aishell3_static_1.1.0"

  # 音频生成的路径，修改成你音频想要保存的路径

  wav_output_dir = "stuido_output"

  # 选择设备[gpu / cpu]，这里以GPU为例子， 

  device = "gpu"

  # 想要生成的文本文档对应文件名

  txt_name = "文本.txt"

  the_main(get_text_dict(name=am_inference_dir,txtname=txt_name))

