import os
import re
import sys
import time
import torch
import queue
import threading
import torchaudio
import numpy as np
import sounddevice as sd
from datetime import datetime
from collections import Counter
from faster_whisper import WhisperModel

# ============================= Logger
class Logger:
    def __init__(self, logfile):
        self.terminal = sys.__stdout__
        self.log = logfile

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ============================= 중복 제거 함수
# 반복 텍스트 필터 함수
def is_repetitive(text: str) -> bool:
    # 1. 문자 반복 검사:
    # 공백을 제거한 후 같은 문자가 5번 이상 반복되면 반복으로 간주
    # 예: "ㅋㅋㅋㅋㅋ", "아아아아아"
    if re.fullmatch(r"(.)\1{4,}", text.replace(" ", "")):
        return True

    # 2. 단어 반복 검사:
    # 공백 기준으로 같은 단어가 연속적으로 5회 이상 반복될 경우 필터링
    # 예: "좋아요 좋아요 좋아요 좋아요 좋아요"
    if re.search(r"\b(\w+)\b(?: \1){4,}", text):
        return True

    # 3. 음절 반복 검사:
    # 같은 음절이 공백 포함 형태로 반복되는 경우 필터링
    # 예: "아 아 아 아 아"
    if re.fullmatch(r"(.)\s*(?:\1\s*){4,}", text):
        return True

    # 4. 단어 빈도 기반 반복 검사:
    # 문장에서 특정 단어가 전체 단어의 30% 이상, 5회 이상 등장할 경우 필터
    words = re.findall(r"\b\w+\b", text)
    total = len(words)
    if total >= 5:
        freq = Counter(words)
        most_common, count = freq.most_common(1)[0]
        if count / total > 0.2 and count >= 5:
            return True
    # 5. n-gram 반복 검사:
    # 2단어, 3단어씩 묶인 문장이 반복되는 경우 필터링
    # 예: "스튜디오에 도착한 스튜디오에 도착한 ..."
    if is_ngram_repetitive(text, n=2):
        return True
    if is_ngram_repetitive(text, n=3):
        return True
    return False

# n-gram 각 문장을 n개씩 조개서 문장 단위로 체크하는 for문과 갯수체크하는 counter로 이뤄어진 O(n)와 nlogn정도
def is_ngram_repetitive(text: str, n=2) -> bool:
    words = text.split()  # 공백 기준 단어 분리  # n-gram 단위 반복 필터 함수
    # n개의 단어를 묶어서 n-gram 리스트 구성, #ex: "스튜디오에 도착한 스튜디오에 도착한" → 3단어 단위 n-gram 반복
    ngrams = [" ".join(words[i: i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return False
    freq = Counter(ngrams)  # n-gram 빈도 측정 (ex: '스튜디오에 도착한': 8회 등)
    most_common, count = freq.most_common(1)[0]  # 가장 많이 나온 n-gram 추출 most_common은 리스트형
    if count >= 5 and count / len(ngrams) > 0.2:  # ({'스튜디오에 도착한': 3, '도착한 후': 1}) 같은 딕셔너리 형태의 튜플로 추출
        return True  # 전체 n-gram 중 특정 문장이 절반 이상 반복되며 5회 이상 등장하면 필터링
    return False

# ============================= STT 처리 메인 함수
def process_audio(model_size, sample_rate, energy_threshold, log_path, device_id, channels, chunk_duration, silence_interval, RECORD_SECONDS, queue_size):

    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)
    sys.stdout.flush()

    model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
    print(f"STT 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    sys.stdout.flush()

    audio_queue = queue.Queue(maxsize=queue_size)
    buffer = []
    buffer_duration = 0.0
    last_spoken_time = time.time()
    silence_marker_printed = False  # 무음 상태 중복 출력 방지
    combined_text = []
    last_output_text = ""
    final_output = ""

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("⚠️", status)
        audio_queue.put(indata.copy())

    def audio_stream():
        with sd.InputStream(device=device_id, channels=channels, samplerate=sample_rate,
                            blocksize=int(sample_rate * chunk_duration), dtype='float32', callback=audio_callback):
            while True:
                time.sleep(0.1)

    threading.Thread(target=audio_stream, daemon=True).start()

    try:
        while True:
            chunk = audio_queue.get()
            audio_tensor = torch.from_numpy(chunk).squeeze()
            energy = torch.mean(torch.abs(audio_tensor)).item()

            # 무음 디버깅용
            # sys.__stdout__.write(f"[DEBUG] chunk_energy={energy:.6f}\n")
            # sys.__stdout__.flush()

            buffer_duration += chunk_duration

            if energy >= energy_threshold:
                if sample_rate != 16000:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, orig_freq=sample_rate, new_freq=16000
                    ).contiguous()
                buffer.append(audio_tensor)
                last_spoken_time = time.time()

            is_forced_flush = buffer_duration >= RECORD_SECONDS
            is_silence_flush = time.time() - last_spoken_time >= silence_interval

            if is_forced_flush or is_silence_flush:
                if buffer:
                    combined_audio = torch.cat(buffer)
                    audio_array = combined_audio.cpu().numpy()
                    segments, _ = model.transcribe(
                        audio_array,
                        language="ko",
                        vad_filter=True,
                        beam_size=1,
                        temperature=0.0
                        )
                    # 중복 텍스트 검사 및 추가
                    for segment in segments:
                        text = segment.text.strip()
                        # 반복 텍스트 필터 적용
                        if not is_repetitive(text):
                            if text != last_output_text:
                                combined_text.append(text)
                                last_output_text = text
                    final_output = " ".join(combined_text).strip()
                    combined_text.clear()

                    if final_output:
                        print(final_output)
                        sys.stdout.flush()
                        silence_marker_printed = False  # 음성 출력 후 다시 "-" 출력 가능하게 초기화

                    final_output = ""
                else:
                    if is_silence_flush and not silence_marker_printed:
                        sys.__stdout__.write("-\n")
                        sys.__stdout__.flush()
                        silence_marker_printed = True

                buffer.clear()
                buffer_duration = 0.0

    except KeyboardInterrupt:
        print(f"\nSTT 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    finally:
        sys.stdout.flush()
        log_file.close()

# ============================= 메인 실행
if __name__ == "__main__":
    MODEL_SIZE = "large-v3"
    DEVICE_ID = 1
    CHANNELS = 1
    ENERGY_GATE_THRESHOLD = 0.001
    RECORD_SECONDS = 15.0
    CHUNK_DURATION = 3.0
    SILENCE_INTERVAL = 3
    queue_size = 20

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    log_dir = os.path.join(desktop, "STT_logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{timestamp}_stt_log.txt")

    try:
        device_info = sd.query_devices(DEVICE_ID, 'input')
        SAMPLE_RATE = int(device_info['default_samplerate'])
    except Exception as e:
        print(f"❌ 마이크 장치 확인 실패: {e}")
        sys.exit(1)

    process_audio(MODEL_SIZE, SAMPLE_RATE, ENERGY_GATE_THRESHOLD, log_path, DEVICE_ID, CHANNELS, CHUNK_DURATION, SILENCE_INTERVAL, RECORD_SECONDS, queue_size)
