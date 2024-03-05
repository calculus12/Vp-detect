from torch import nn
import torch.nn.functional as F
from datetime import datetime

# ---------------------------------------------------
# Model Serving

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    

import torch
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
import gluonnlp as nlp

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
tok = tokenizer.tokenize
max_len = 64

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

transform = nlp.data.BERTSentenceTransform(
            tok, max_seq_length=max_len, vocab=vocab, pad=True, pair=False)

# 모델 불러오기
# model = BERTClassifier(bertmodel,  dr_rate=0.4).to(device)
# model.load_state_dict(torch.load('/content/drive/MyDrive/dl-team-project/model_state_dict.pt',map_location=torch.device('cpu')))
model = BERTClassifier(bertmodel,  dr_rate=0.4).to(device)
model.load_state_dict(torch.load('./model_state_dict.pt',map_location=torch.device('cpu')))
model.eval()

def predict_vp(input_text:str):
    # tokenize
    transformed_text = transform([input_text])
    token_ids = torch.Tensor([transformed_text[0]]).long().to(device)
    valid_length = torch.tensor(transformed_text[1]).unsqueeze(0)
    valid_length = valid_length.to(device)
    segment_ids = torch.Tensor([transformed_text[2]]).long().to(device)

    # predict
    with torch.no_grad():
        out = model(token_ids, valid_length, segment_ids)
        pred = out.detach()
        pred = F.softmax(pred)
        pred = pred[:, 1].cpu().numpy().tolist()
        return pred[0]


# Model Serving
# --------------------------------------------------------------------------------------------
# Server

import shutil
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import base64
import os



app = FastAPI()

class AudioData(BaseModel):
    data: str 


# cloud speech-to-text v2
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.cloud import storage

def get_transcribe(audio_data:str)->str:
    # Instantiates a client
    client = SpeechClient()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["ko-KR"],
        model="long",
    )

    request = cloud_speech.RecognizeRequest(
    recognizer=f"projects/solar-attic-406607/locations/global/recognizers/_",
    config=config,
    content=audio_data,
    )

    # Detects speech in the audio file
    response = client.recognize(request=request)

    result_transcribe = ""
    for result in response.results:
        result_transcribe += result.alternatives[0].transcript
        result_transcribe += "\n"
    
    return result_transcribe


BUCKET_NAME = "stt-audio-dl"
PROJECT_ID = 'solar-attic-406607'
DESTINATION_BLOB_NAME = 'vp-detect-audio'

def get_blob_name_with_timestamp()->str:
    timestamp = datetime.timestamp(datetime.now())
    formatted_time = datetime.fromtimestamp(timestamp).strftime('-%y%m%d-%H:%M:%S')
    return DESTINATION_BLOB_NAME + formatted_time + '.raw'

def get_gcs_uri_from_blob_name(blob_name: str)->str:
    return 'gs://' + BUCKET_NAME + '/' + blob_name

def upload_to_gcs(contents, blob_name:str):
    """Uploads a audio file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    blob.upload_from_string(contents)

# 1분 이상의 오디오 파일의 stt 가능
def get_transcribe_v2(gcs_uri:str)->str:
    client = SpeechClient()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["ko-KR"],
        model="long",
    )

    file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_",
        config=config,
        files=[file_metadata],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
    )

    # Transcribes the audio into text
    operation = client.batch_recognize(request=request)
    response = operation.result(timeout=120)

    result_transcribe = ""
    for result in response.results[gcs_uri].transcript.results:
        result_transcribe += result.alternatives[0].transcript
        result_transcribe += '\n'
    
    return result_transcribe


@app.post("/detect-phishing")
async def detect_phishing(audio_data: AudioData):
    decoded_data = base64.b64decode(audio_data.data)

    blob_name = get_blob_name_with_timestamp()
    gcs_uri = get_gcs_uri_from_blob_name(blob_name)

    upload_to_gcs(decoded_data, blob_name)
    transcript = get_transcribe_v2(gcs_uri)
    res = predict_vp(transcript)
    return {"transcript":transcript, "result": str(res)}
    


import uvicorn
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
