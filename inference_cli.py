import os
from webui import utils, audio, config
import webui
import sys
import vc_infer_pipeline
import ast
from scipy.io import wavfile

CWD = webui.get_cwd()

state = {
    'f0_up_key': 0, 
    'f0_method': ['crepe'], 
    'f0_autotune': False, 
    'merge_type': 'median', 
    'index_rate': 0.75, 
    'filter_radius': 3, 
    'resample_sr': 0, 
    'rms_mix_rate': 0.2, 
    'protect': 0.2
}

if __name__=="__main__":
	#인자
	#modelList : 모델 목록
	#vocalList : 음성 목록

	#1 : 모델 경로(따옴표 붙어서 하나로)
	#2 : 입력 오디오 경로(wav만)
	#3 : f0_up_key (-12 ~ 12 사이의 정수)
    #4 : f0_method (문자열 리스트)
    #5 : f0_autotune (True, False)
    #6 : merge_type ('mean', 'median')
    #7 : index_rate (0.00 ~ 1.00, 0.05 단위로 조정)
    #8 : filter_radius (0 ~ 7 사이의 정수)
    #9 : resample_sr (0, 16000, 24000, 22050, 40000, 44100, 48000)
    #10 : rms_mix_rate (0.00 ~ 1.00, 0.05 단위로 조정)
    #11 : protect (0.00 ~ 0.50, 0.01 단위로 조정)

	if len(sys.argv) != 13:
		print("Error : argument count")
		exit(-1)

	modelPath = sys.argv[1]
	inputAudioPath = sys.argv[2]

	state = {
    	'f0_up_key': int(sys.argv[3]),
    	'f0_method': ast.literal_eval(sys.argv[4]),
    	'f0_autotune': sys.argv[5] == "True",
		'merge_type': sys.argv[6],
    	'index_rate': float(sys.argv[7]),
    	'filter_radius': int(sys.argv[8]), 
    	'resample_sr': int(sys.argv[9]), 
    	'rms_mix_rate': float(sys.argv[10]), 
    	'protect': float(sys.argv[11])
	}


	"""
		기본값
		f0_up_key=0,
        f0_method=["rmvpe"],
        f0_autotune=False,
        merge_type="median",
        index_rate=.75,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=.2,
        protect=0.2,
	"""

	print(modelPath)
	print(inputAudioPath)
	print(state)
	
	#모델 불러오기
	device = utils.get_optimal_torch_device()
	LoadedModel = vc_infer_pipeline.get_vc(modelPath, config=config, device=device)
	utils.gc_collect()

	wavData = None

	#음성 가져오기
	with open(inputAudioPath, 'rb') as wavFile:
		wavData = wavFile.read()
	
	input_audio = audio.bytes_to_audio(wavData)
	original_sr = input_audio[1]
	input_vocals = audio.remix_audio(input_audio, norm=True, to_int16=True, to_mono=True)
	input_audio_name = os.path.basename(inputAudioPath)

	#변환
	output_audio = None
	try:
		output_audio = vc_infer_pipeline.vc_single(input_audio = input_vocals, **LoadedModel, **state)
	except Exception as e:
		print(e)
		exit(-1)

	modelName = os.path.basename(modelPath)

	output_path = sys.argv[12] + input_audio_name[:input_audio_name.rindex('.')] + " (" + modelName[:modelName.rindex('.')] + ").wav"

	wavfile.write(output_path, output_audio[1], output_audio[0])

	print("End!")