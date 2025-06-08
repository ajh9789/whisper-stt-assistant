from datetime import datetime
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from faster_whisper import WhisperModel
from multiprocessing import Process, Queue
import time

# =============================
# 🎯 설정 값
# =============================

MODEL_SIZE = "large-v3"
DEVICE_ID = 1
RECORD_SECONDS = 10
CHANNELS = 1
ENERGY_GATE_THRESHOLD = 0.0007

# =============================
# 🎧 디바이스 및 모델 초기화
# =============================

device_info = sd.query_devices(DEVICE_ID, 'input')
SAMPLE_RATE = int(device_info['default_samplerate'])
blocksize = int(SAMPLE_RATE * 2)  # n초 단위 블록

# =============================
# 🔁 오디오 캡처 콜백
# =============================

def audio_callback(indata, frames, time_info, status, queue):
    if status:
        print("⚠️", status)
    queue.put(indata.copy())

def start_recording(queue: Queue):
    def callback(indata, frames, time_info, status):
        audio_callback(indata, frames, time_info, status, queue)

    with sd.InputStream(
        device=DEVICE_ID,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=blocksize,
        dtype='float32',
        callback=callback
    ):
        while True:
            time.sleep(0.1)

# =============================
# 🧠 STT 처리 프로세스
# =============================

def process_audio(queue: Queue):
    model = WhisperModel(MODEL_SIZE, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
    print(f"🚀 STT 모델 로드 완료")

    audio_chunks = []
    total_frames = 0
    target_frames = SAMPLE_RATE * RECORD_SECONDS
    last_spoken_time = time.time()
    silence_interval = 5  # 무음 경고 기준 초

    while True:
        chunk = queue.get()
        audio_chunks.append(chunk)
        total_frames += chunk.shape[0]

        if total_frames >= target_frames:
            audio = np.concatenate(audio_chunks, axis=0)
            audio_chunks.clear()
            total_frames = 0

            audio_tensor = torch.from_numpy(audio).squeeze()

            # 무음 필터
            if torch.mean(torch.abs(audio_tensor)) < ENERGY_GATE_THRESHOLD:
                # 무음 시간 체크
                if time.time() - last_spoken_time >= silence_interval:
                    print(f"🔇 [무음 상태: {silence_interval}초 이상]")
                    last_spoken_time = time.time()
                continue

            # 리샘플링
            if SAMPLE_RATE != 16000:
                audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=SAMPLE_RATE, new_freq=16000).contiguous()

            audio_array = audio_tensor.cpu().numpy()

            # STT 실행
            segments, _ = model.transcribe(
                audio_array,
                language="ko",
                vad_filter=True,
                beam_size=1,
                temperature=0.0
            )

            for segment in segments:
                print(f"{segment.text.strip()}")
                last_spoken_time = time.time()  # 발화 시간 갱신

# =============================
# 🎯 실행
# =============================

if __name__ == "__main__":
    print(f"\n🎙️ 디바이스 {DEVICE_ID}: {device_info['name']} ({SAMPLE_RATE} Hz)")
    print(f"✅ FasterWhisper {MODEL_SIZE} 모델 로드 예정")
    print(f"🔄 {RECORD_SECONDS}초마다 자동 녹음 + 변환 시작 (Ctrl+C로 중지)\n")

    queue = Queue(maxsize=10)

    recorder = Process(target=start_recording, args=(queue,))
    stt_worker = Process(target=process_audio, args=(queue,))

    recorder.start()
    stt_worker.start()

    try:
        recorder.join()
        stt_worker.join()
    except KeyboardInterrupt:
        recorder.terminate()
        stt_worker.terminate()
        print("\n🛑 프로그램 종료됨.")

