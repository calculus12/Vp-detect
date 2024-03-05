import streamlit as st
from io import BytesIO
import requests
import base64
import json
import os
from pydub import AudioSegment
from datetime import datetime

SERVER_ENDPOINT = "http://34.22.69.228:8000/detect-phishing"
LOCAL_ENDPOINT = "http://localhost:8000/detect-phishing"
CONVERTED_WAV_PATH = 'converted_wav/temp'

def temp_wav_file_path()->str:
    timestamp = datetime.timestamp(datetime.now())
    formatted_time = datetime.fromtimestamp(timestamp).strftime('-%y%m%d-%H:%M:%S')
    path = CONVERTED_WAV_PATH + formatted_time + '.wav'
    return path

def delete_file(file_path):
    # 임시로 저장한 WAV 파일 삭제용 함수
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"Error deleting converted wav file: {e}")

def bytes_base64_econde(file_bytes):
    # 파일을 바이너리 모드로 열어서 읽고 base64로 인코딩
    base64_encoded_data = base64.b64encode(file_bytes).decode('utf-8')
    return base64_encoded_data

def convert_m4a_to_wav(m4a_bytes, path):
    audio = AudioSegment.from_file(BytesIO(m4a_bytes), format="m4a")
    file_handle = audio.export(path, format='wav')
    with open(path, "rb") as file:
        wav_bytes = file.read()
    return wav_bytes

# Streamlit 앱 제목 설정
st.title("Korean Voice Phishing Detection with koBERT and Google STT")

# 파일 업로드 위젯
uploaded_file = st.file_uploader("Choose a file", type=["wav", "m4a", "mp3"])

# 파일이 업로드되었을 때 처리
if uploaded_file is not None:


    file_extension = uploaded_file.name.split(".")[-1].lower()
    # 확장자가 m4a인지 확인
    if file_extension == "m4a":
        m4a_bytes = uploaded_file.read()
        # m4a를 wav로 변환
        temp_path = temp_wav_file_path()
        wav_bytes = convert_m4a_to_wav(m4a_bytes, temp_path)
        encoded_data = bytes_base64_econde(wav_bytes)
        delete_file(temp_path)
    else:
        uploaded_file_bytes = uploaded_file.read()
        encoded_data = bytes_base64_econde(uploaded_file_bytes)

    # 바이너리 데이터를 JSON 형식으로 변환
    json_payload = {"data": encoded_data}
    headers = {"Content-Type": "application/json"}
    # 서버에 HTTP POST 요청 보내기
    server_url = LOCAL_ENDPOINT

    # 서버 응답 확인
    try:
        response = requests.post(server_url, json=json_payload, headers=headers)
        # HTTP 요청이 성공하면 확률을 큰 글씨로 표시
        if response.status_code == 200:
            response_json = response.json()
            transcribe = response_json["transcript"]
            probability = float(response_json["result"]) * float(100)
            formatted_transcribe = f"<h3>Call record transcribe</h3>{transcribe}"
            formatted_prob = f"<h3>Probability that your call record was Voice Phishing</h3><h1>{probability:.2f}%</h1>"
            st.markdown(formatted_transcribe, unsafe_allow_html=True)
            st.divider()
            st.markdown(formatted_prob, unsafe_allow_html=True)
        else:
            st.error("Failed to get result from the server.")
            print(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Error during the HTTP request: {e}")

