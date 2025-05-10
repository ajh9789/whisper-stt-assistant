from datetime import datetime
import time
import numpy as np
import sounddevice as sd
import torch
import whisper
from scipy.signal import resample_poly

# =============================
# 🎯 설정 값
# =============================

MODEL_SIZE = "medium"                  # small → 빠름 / medium → 정확도 / large-v3 → 무거움
DEVICE_ID = 14                         # 🎙️ 사용할 microphone device index
RECORD_SECONDS = 5                    # 🎙️ 녹음 길이 (초)
CHANNELS = 1                           # mono
ENERGY_GATE_THRESHOLD = 0.001          # ✅ 민감도: 낮으면 민감 ↑, 높으면 둔감 ↓

# =============================
# 🎧 디바이스 및 whisper 모델 초기화
# =============================

device_info = sd.query_devices(DEVICE_ID, 'input')
SAMPLE_RATE = int(device_info['default_samplerate'])    # 🎙️ 마이크 샘플레이트

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(MODEL_SIZE, device=device)  # ✅ 모델 로드

print(f"\n🎙️ 디바이스 {DEVICE_ID}: {device_info['name']} ({SAMPLE_RATE} Hz)")
print(f"✅ Whisper {MODEL_SIZE} 모델 로드 완료 (device: {device})")
print(f"🔄 {RECORD_SECONDS}초마다 자동 녹음 + 변환 시작 (Ctrl+C로 중지)\n")

# =============================
# 🎙️ STT 함수
# =============================

def record_and_transcribe():
    """
    🎙️ n초 녹음 → whisper STT 변환
    """
    # print(f"🎙️ {RECORD_SECONDS}초 녹음 중...")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype='float32', device=DEVICE_ID)
    sd.wait()      # ✅ 녹음 완료 대기
    audio = np.squeeze(audio)

    # ✅ 민감도 필터: 평균 진폭이 너무 낮으면 무시
    avg_amplitude = np.mean(np.abs(audio))
    if avg_amplitude < ENERGY_GATE_THRESHOLD:
        return

    # ✅ whisper 입력용 변환 (16kHz, float32)
    if SAMPLE_RATE != 16000:
        audio = resample_poly(audio, up=16000, down=SAMPLE_RATE)

    # print("🔎 whisper 변환 중...")

    # ✅ whisper STT
    result = model.transcribe(
        audio,
        language="ko",
        fp16=True,                           # ✅ GPU 사용 시 속도 향상
        temperature=0,
        condition_on_previous_text=False
    )

    print(f"[{datetime.now().strftime('%H:%M:%S')}]:{result['text']}")

# =============================
# 🎯 무한 반복
# =============================

if __name__ == "__main__":
    try:
        while True:
            record_and_transcribe()
            # time.sleep(0.1)     # ✅ 약간의 대기 (연속 녹음 안정화)
    except KeyboardInterrupt:
        print("\n🛑 프로그램 종료됨.")