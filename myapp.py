from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import pickle as p
import requests
import flask
import json
import os


import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import wave
from IPython.display import Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

myapp = Flask(__name__)

@myapp.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@myapp.route('/',methods=['POST'])

# def upload():
#      if flask.request.method == "POST":
#          audiofiles = flask.request.files.getlist("audiofile")
#          for audiofile in audiofiles:
#              audio_path = "./voicenotes/" + audiofile.filename
#              audiofile.save(audio_path)

def predict():
    audiofile = request.files['audiofile']
    audio_path = "./voicenotes/" + audiofile.filename
    audiofile.save(audio_path)

    path = "./voicenotes"
    directory = os.fsencode(path)

    # Function to read audio files
    # def read_text_file(file_path):
    #     with open(file_path, 'r') as f:
    #         print(f.read())

    audiolist = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"): # Check whether file is in wav format or not
            audiolist.append(f"{filename}")

    #Generate transcrition from XLS-R model
    transcriptions = []
    for voice in audiolist:
        data = wavfile.read(voice)
        framerate = data[0]
        sounddata = data[1]
        time = np.arange(0,len(sounddata))/framerate
        input_audio, _ = librosa.load(voice, sr=16000)

        input_values = tokenizer(input_audio, return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        transcriptions.append(transcription)

        with open('data.txt', 'w') as f:
            for line in transcriptions:
                f.write(line)
                f.write('\n')

    def readFile(fileName):
        """
        This function will read the text files passed & return the list
        """
        fileObj = open(fileName, "r") #opens the file in read mode
        words = fileObj.read().splitlines() #puts the file into a list
        fileObj.close()
        return words

    df = readFile('data.txt')

    embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    corpus_embeddings = embedder.encode(df)

    num_clusters = 1
    clustering_model = KMeans(n_clusters=num_clusters) # Define kmeans model
    clustering_model.fit(corpus_embeddings) # Fit the embedding with kmeans clustering.
    cluster_assignment = clustering_model.labels_ # Get the cluster id assigned to each news headline.

    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(df[sentence_id])
    for i, cluster in enumerate(clustered_sentences):
        print("Cluster ", i+1)
        print(cluster)
        print("")
        #category = cluster_assignment[i]

    return render_template('index.html', prediction=transcription)

if __name__ == '__main__':
    myapp.run(debug=True,port=3000)
