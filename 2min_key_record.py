from datetime import datetime
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from scipy.signal import resample_poly
import keyboard
import queue
import time

# =============================
# 🎯 설정 값
# =============================

MODEL_SIZE = "large-v3"               # whisper 모델 크기 (small, medium, large 등)
DEVICE_ID = 1                         # 🎙️ 사용할 microphone device index
CHANNELS = 1                           # mono (단일 채널)
ENERGY_GATE_THRESHOLD = 0.0008          # ✅ 감도: 너무 조용한 녹음은 무시
MAX_RECORD_SECONDS = 120                # ✅ 최대 녹음 시간 3070은 2분넘어가면 변환시간이 너무오래걸림

# =============================
# 🎧 디바이스 및 모델 초기화
# =============================

device_info = sd.query_devices(DEVICE_ID, 'input')
SAMPLE_RATE = int(device_info['default_samplerate'])   # 🎙️ 마이크 기본 샘플레이트

# whisper 모델 로드 (GPU 사용 가능 시 GPU 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(MODEL_SIZE, device=device, compute_type="float16")

print(f"\n🎙️ 디바이스 {DEVICE_ID}: {device_info['name']} ({SAMPLE_RATE} Hz)")
print(f"✅ Whisper {MODEL_SIZE} 모델 로드 완료 (device: {device})")
print("▶ scroll → 녹음 시작, pause → 중지 + 변환 (Ctrl+C 종료)\n")

# =============================
# 🎙️ 전역 변수 정의
# =============================

recording = False               # 현재 녹음 중인지 상태 표시
audio_queue = queue.Queue()     # 녹음 데이터를 담는 버퍼 (stream 콜백 → main으로 전달)

# =============================
# 🎙️ 오디오 스트림 콜백 함수
# =============================

def audio_callback(indata, frames, time_info, status):
    """
    🎙️ stream이 켜져 있는 동안 계속 호출됨
    → 녹음 데이터를 indata로 받아서 queue에 저장
    """
    if recording:
        audio_queue.put(indata.copy())  # 원본 데이터 복사 후 queue에 저장

# =============================
# 🎙️ 녹음 데이터 → 텍스트 변환 함수
# =============================

def process_audio():
    audio_chunks = []
    total_samples = 0
    max_samples = SAMPLE_RATE * MAX_RECORD_SECONDS

    while not audio_queue.empty() and total_samples < max_samples:
        chunk = audio_queue.get()
        audio_chunks.append(chunk)
        total_samples += chunk.shape[0]

    if not audio_chunks:
        print("⚠️ 녹음 데이터 없음.")
        return

    audio = np.concatenate(audio_chunks, axis=0).flatten()

    if np.mean(np.abs(audio)) < ENERGY_GATE_THRESHOLD:
        print("⚠️ 너무 조용해서 무시됨.")
        return

    if SAMPLE_RATE != 16000:
        audio = resample_poly(audio, up=16000, down=SAMPLE_RATE)

    # ✅ fast-whisper 스타일로 수정
    segments, info = model.transcribe(
        audio.astype(np.float32),
        language="ko",
        vad_filter=True,
        beam_size=1,
        temperature=0.0
    )

    print("[답변]:", end=" ")
    for segment in segments:
        print(segment.text.strip(), end=" ")
    print()


# =============================
# 🎯 메인 실행 루프
# =============================

if __name__ == "__main__":
    try:
        # 🎙️ stream 초기화 → 키 누를 때만 start/stop
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            device=DEVICE_ID,
            channels=CHANNELS,
            callback=audio_callback   # stream에서 계속 호출할 콜백 등록
        )

        # 🎯 무한 루프: 키 입력 대기
        while True:
            # ✅ scroll lock 키 누르면 → stream 시작
            if keyboard.is_pressed('scroll lock') and not recording:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]🎬 녹음 시작 (scroll)")
                audio_queue.queue.clear()    # 이전 버퍼 초기화 (깨끗한 시작)
                stream.start()               # stream 시작 → 콜백으로 audio_queue 채움
                recording = True
                time.sleep(0.5)              # 키 중복 입력 방지 (디바운스)

            # ✅ pause 키 누르면 → stream 정지 + STT 변환
            if keyboard.is_pressed('pause') and recording:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]🛑 녹음 중지 + 변환 시작 (pause)")
                stream.stop()                # stream 멈춤 → 더 이상 audio_queue에 안 넣음
                recording = False
                process_audio()              # queue에 남은 데이터 → STT 수행
                time.sleep(0.5)              # 중복 입력 방지

    except KeyboardInterrupt:
        # ✅ Ctrl+C 누르면 프로그램 종료
        print("\n🛑 프로그램 종료됨.")
        if recording:
            stream.stop()
