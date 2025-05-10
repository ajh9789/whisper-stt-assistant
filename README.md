# WhisperSTT (RTX 3070 + fp16)

## 🎙️ Microphone recording + Whisper STT batch assistant 
## 사용자의 음성을 자동으로 텍스트로 바꿔주는 면접 연습 및 피드백용 project입니다.
> 📋 본 프로젝트의 문서 작성 및 코드 정리는 ChatGPT를 도구로 참고/보조하여 진행되었습니다.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![ChatGPT](https://img.shields.io/badge/ChatGPT-Assistant-10A37F?logo=openai&logoColor=white)
![OpenAI Whisper](https://img.shields.io/badge/OpenAI-Whisper-6e00ff?logo=openai&logoColor=white)
![Sounddevice](https://img.shields.io/badge/sounddevice-Audio-blueviolet)
![NVIDIA](https://img.shields.io/badge/NVIDIA-RTX%203070-76B900?logo=nvidia&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-10%2B-0078D6?logo=windows&logoColor=white)
![PyCharm](https://img.shields.io/badge/PyCharm-Professional-green?logo=jetbrains&logoColor=white)
![Git](https://img.shields.io/badge/Git-Version%20Control-orange?logo=git&logoColor=white)

## 🛠️ 기술 스택

- **언어**: Python 3.9 이상
- **라이브러리**: 
  - sounddevice
  - OpenAI Whisper (MIT 라이선스)
- **개발 환경**: 
  - Windows10
  - Python 가상환경 (venv)
- **개발 도구**:
  - Git
  - PyCharm
  - ChatGPT (문서 및 코드 보조 도구)
----
## 💻 테스트 환경

- NVIDIA RTX 3070
- AMD Ryzen 5 5600X
- 32GB RAM
---

## 📥 설치 방법

1️⃣ Python 3.9 이상 설치  
2️⃣ 가상환경 생성  

- python -m venv venv  
- venv\Scripts\activate      (Windows)  
- source venv/bin/activate   (Linux/Mac)

3️⃣ 필수 패키지 설치  

- pip install -r requirements.txt

---

## 🎙️ 마이크 DEVICE_ID 설정 방법

assistant 사용 전, 본인의 마이크 장치 번호 (`DEVICE_ID`)를 확인해야 합니다.

### ✅ 1️⃣ 사용 가능한 마이크 목록 확인

Python에서 아래 코드를 실행하세요.

```python
# 사용 가능한 마이크 목록 확인 예시
import sounddevice as sd
print(sd.query_devices())

# 예시 출력
# 0: Microsoft Sound Mapper - Input
# 1: 마이크 (USB Audio Device)
# 2: 스테레오 믹스 (Realtek Audio)
# ...
```


---

### ✅ 2️⃣ 원하는 장치 번호 선택

원하는 마이크 장치의 번호를 코드에 지정합니다.

DEVICE_ID = 1

👉 예시: 마이크 (USB Audio Device)를 쓰고 싶다면 DEVICE_ID = 1

> ⚠️ 주의: 잘못된 값을 입력하면 "Invalid device" 오류가 발생합니다.

---

## 🎛️ 사용 방법

### 📋 버전 1: 5초 주기 자동 인식 (5s_time_record.py)

- 프로그램 실행 후, 종료 전까지 5초마다 자동으로 음성을 인식하고 텍스트를 출력합니다.

### 📋 버전 2: 마지막 2분을 scroll/pause 키로 녹음 제어 (2min_key_record.py)

- 프로그램 실행 후, 마지막 2분을 기준으로 음성을 인식하고 텍스트로 출력합니다.
- scroll : 녹음 시작 (키보드 오른쪽 위에 있는거!) 
- pause : 녹음 중지 + Whisper STT 변환 (키보드 오른쪽 위에 있는거!) 
- Ctrl+C : 프로그램 종료  

---
## ✅ 1. 모델별 5초 음성 변환 속도 (RTX 3070 + fp16 기준)

| 모델                | 변환 시간  | 비고                             | 평점  |
|---------------------|------------|----------------------------------|------|
| tiny                | 0.2~0.3초  | 초경량 (실시간 assistant용)       |      |
| base                | 0.3~0.4초  | 가벼운 assistant                 |      |
| small               | 0.5~0.8초  | 실용적 + fast                    | ⭐⭐   |
| medium              | 1.2~1.8초  | 개인 assistant 권장               | ⭐⭐⭐  |
| large-v2 / large-v3 | 2.5~3.5초  | 최고 정확도, 면접 assistant 추천 | ⭐    |

## ✅ 2. large-v3 권장 녹음 시간 (RTX 3070 기준 + fp16 기준)

| 권장         | 비고        | 평점  |
|--------------|-------------|------|
| 30~60초      | best 안정성 | ⭐⭐   |
| 90초         | 실용적      | ⭐⭐⭐  |
| 120초        | 한계        | ⭐    |
| 150초 이상   | 절대 비추천 |      |



> ⚠️ **주의:** Whisper 모델은 입력 audio 길이에 따라 처리 속도가 비선형적으로 증가합니다.  
> RTX 3070 + fp16 기준으로 **2분(120초) 이하에서는 안정적**이나,  
> **2분을 초과하면 변환 시간이 급격히 느려지고 (최대 2~3배 지연)**,  
> assistant 실시간성/안정성이 크게 떨어질 수 있습니다.  
> **가능하면 120초 이하로 녹음**하는 것을 강력하게 권장합니다.

---


## ✅ 기타

- venv, .idea, 등등 파일들은 `.gitignore`로 자동 제외됩니다.  

 ### 파일구조

```plaintext
WhisperSTT_Assistant/
├── .gitignore
├── README.md
├── requirements.txt
├── 5s_time_record.py
├── 2min_key_record.py
├── venv/ → (gitignore로 제외됨)
```

---

## 📢 참고
본 프로젝트는 OpenAI Whisper 모델 및 sounddevice 라이브러리를 활용합니다.
Whisper is an open-source model released by OpenAI (MIT License).
---