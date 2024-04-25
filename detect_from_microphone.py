# Main:
# Module xử lý wakeword từ người dùng 
# để khởi động hệ thống
import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse
from utils.beep import playBeep
import os

# Phân tích đối số đầu vào
parser=argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="Có bao nhiêu âm thanh (về số lượng mẫu) để dự đoán cùng một lúc",
    type=int,
    default=1280,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default='tflite',
    required=False
)

args=parser.parse_args()

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load pre-trained openwakeword models
if args.model_path != "":
    owwModel = Model(wakeword_models=[args.model_path], inference_framework=args.inference_framework, enable_speex_noise_suppression=True)
else:
    owwModel = Model(inference_framework=args.inference_framework, enable_speex_noise_suppression=True)

n_models = len(owwModel.models.keys())

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    # Generate output string header
    print("\n\n")
    print("#"*100)
    print("Đang lắng nghe từ kích hoạt...")
    print("#"*100)
    print("\n"*(n_models*3))

    while True:
        # Lấy âm thanh
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Nguồn cấp dữ liệu cho mô hình openWakeWord
        prediction = owwModel.predict(audio)
        # if prediction == 0.65:
            # playBeep(os.path.join(os.path.dirname(__file__), 'audio', 'activation.wav'), audio)

        # Tiêu đề cột
        n_spaces = 16
        output_string_header = """
            Tên mô hình         | Điểm | Trạng thái WakeWord
            ----------------------------------------------------
            """

        for mdl in owwModel.prediction_buffer.keys():
            # Add scores in formatted table
            scores = list(owwModel.prediction_buffer[mdl])
            curr_score = format(scores[-1], '.20f').replace("-", "")
            print(curr_score)
            # if curr_score >= "0.65":
                # playBeep(os.path.join(os.path.dirname(__file__), 'audio', 'activation.wav'), audio)
            output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Đã phát hiện WakeWord!"}
            """

        # Print results table
        print("\033[F"*(4*n_models+1))
        print(output_string_header, "                             ", end='\r')

    # except KeyBoardInterrupt:
        