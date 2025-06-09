import os
import re
import sys
import time
import torch
import torchaudio
import numpy as np
import sounddevice as sd
from datetime import datetime
from faster_whisper import WhisperModel
from multiprocessing import Process, Queue

# ============================= Logger
# Logger 클래스 수정
class Logger:
    def __init__(self, logfile):
        self.terminal = sys.__stdout__
        self.log = logfile

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 💡 여기 추가!

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ============================= 오디오 콜백
def audio_callback(indata, frames, time_info, status, queue):
    if status:
        print("⚠️", status)
    queue.put(indata.copy())

# ============================= 오디오 녹음
# ============================= 오디오 녹음
def start_recording(queue: Queue, device_id, channels, sample_rate, blocksize):
    def callback(indata, frames, time_info, status):
        audio_callback(indata, frames, time_info, status, queue)

    try:
        with sd.InputStream(device=device_id, channels=channels, samplerate=sample_rate,
                            blocksize=blocksize, dtype='float32', callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass  # ✅ Ctrl+C 시 콘솔 예외 메시지 없이 종료

# ============================= 중복 단어/문장 제거
def remove_repeated_words(text):
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

def is_duplicate_text(text, last_text):
    return text == last_text

# ============================= STT 처리
def process_audio(queue: Queue, model_size, sample_rate, energy_threshold, log_path):

    log_file = open(log_path, "a", encoding="utf-8")

    # ✅ Logger 적용 전에 표준 출력 인코딩 설정
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)

    sys.stdout.flush()

    model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
    start_time = datetime.now()
    print(f"STT 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    sys.stdout.flush()

    silence_interval = 6
    last_spoken_time = time.time()
    last_output_text = ""

    buffer = []
    buffer_duration = 0.0
    chunk_duration = 2.0
    try:
        while True:
            chunk = queue.get()
            audio_tensor = torch.from_numpy(chunk).squeeze()
            energy = torch.mean(torch.abs(audio_tensor)).item()

            # sys.__stdout__.write(f"[DEBUG] chunk_energy={energy:.6f}\n")
            # sys.__stdout__.flush()

            if energy < energy_threshold:
                if time.time() - last_spoken_time >= silence_interval:
                    sys.__stdout__.write("-\n")
                    sys.__stdout__.flush()
                    last_spoken_time = time.time()
                continue

            if sample_rate != 16000:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, orig_freq=sample_rate, new_freq=16000
                ).contiguous()

            buffer.append(audio_tensor)
            buffer_duration += chunk_duration

            if buffer_duration >= 10.0:
                combined_audio = torch.cat(buffer)
                audio_array = combined_audio.cpu().numpy()

                segments, _ = model.transcribe(audio_array, language="ko", vad_filter=False, beam_size=1, temperature=0.0)
                for segment in segments:
                    clean_text = remove_repeated_words(segment.text.strip())
                    if not is_duplicate_text(clean_text, last_output_text):
                        print(clean_text)
                        sys.stdout.flush()
                        last_output_text = clean_text
                        last_spoken_time = time.time()

                buffer.clear()
                buffer_duration = 0.0
    except KeyboardInterrupt:
        end_time = datetime.now()
        print(f"\nSTT 종료: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    finally:
        sys.stdout.flush()
        log_file.close()

# ============================= 메인 실행
if __name__ == "__main__":
    MODEL_SIZE = "large-v3"
    DEVICE_ID = 1
    RECORD_SECONDS = 10
    CHANNELS = 1
    ENERGY_GATE_THRESHOLD = 0.009

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    log_dir = os.path.join(desktop, "STT_logs")
    os.makedirs(log_dir, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{timestamp}_stt_log.txt")
    log_file = open(log_path, "w", encoding="utf-8-sig")  # BOM 포함으로 수정

    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)

    try:
        device_info = sd.query_devices(DEVICE_ID, 'input')
        SAMPLE_RATE = int(device_info['default_samplerate'])
        blocksize = int(SAMPLE_RATE * 2)
    except Exception as e:
        print(f"❌ 마이크 장치 확인 실패: {e}")
        sys.exit(1)

    sys.stdout.flush()

    queue = Queue(maxsize=10)
    recorder = Process(target=start_recording, args=(queue, DEVICE_ID, CHANNELS, SAMPLE_RATE, blocksize))
    stt_worker = Process(target=process_audio, args=(queue, MODEL_SIZE, SAMPLE_RATE, ENERGY_GATE_THRESHOLD, log_path))

    sys.stdout.flush()

    recorder.start()
    stt_worker.start()

    try:
        recorder.join()
        stt_worker.join()
    except KeyboardInterrupt:
        sys.stdout = sys.__stdout__   # ✅ 로그 출력 중단
        sys.stderr = sys.__stderr__
        print("\n__Ctrl+C로 종료됨.__")  # ✅ 콘솔에만 출력
    finally:
        end_time = datetime.now()
        sys.stdout.flush()
        log_file.close()
        print(f"__로그 저장 위치: {log_path}__\n\n")
