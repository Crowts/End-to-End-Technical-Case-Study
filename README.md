# AI driven tool for Voice activated Banking Services
A simple deployment of an AI driven model for the categorisation of customer requests to the relevant business department for fulfilment.

## Problem Description
 The business wants to implement banking services where customers can request services via a WhatsApp voice note. The business wants to establish in real-time what service the customer is requesting so they can trigger some upstream process. The requirement is to build a model that will ingest the audio recording and determine what service fulfilment category they belong to.  

## AI Functionality
The AI functionality is comprised of two parts. The first uses a pre-trained [XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-1b) model to transcribe audio files and the second uses [Sentence-transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeddings to generate representations of the transcription

<br>

# Requirements
The following requirements are necessary to run this project. If you are working in a virtual environment locally, you can do
```sh
pip install -r requirements.txt
```
<br/>
scikit-learn
<br/>
numpy
<br/>
scipy
<br/>
nltk
<br/>
torch
<br/>
torchaudio
<br/>
librosa
<br/>
soundfile
<br/>
IPython
<br/>
transformers
<br/>
huggingface_hub
<br/>

## To run locally:
1. Execute `git clone https://github.com/Crowts/End-to-End-Technical-Case-Study.git`
2. Open command prompt (or terminal) and navigate to the directory 'End-to-End-Technical-Case-Study'
3. Execute
```sh
python myapp.py
```
4. The server should startup and the API can be accessed on http://127.0.0.1:3000/ in your browser.
5. Upload a file from the audiofiles folder and click `Predict fulfilment category`

## Some challenges
- The key challenge with this PoC is throughput and latency. The model XLS-R model takes at least a couple of seconds to generate a transcription for each input and this may impact service time scale. A cloud deployment on a cluster might remedy this problem.<br/>
- The audio data may be a challenge depending on file format. The current Implementation takes only wave files and recordings that have a different file format like MPEG Audio will need to be handled differently.<br/>
- The service does not yet have

## References
[Huggingface](https://huggingface.co/)
<br/>
[Sentence-transformers](https://pypi.org/project/sentence-transformers/) <br/>
[flask](https://flask.palletsprojects.com/en/2.2.x/quickstart/)
