# AI driven tool for Voice activated Banking Services

## Problem Description
A simple deployment of an AI driven model for the categorisation of customer requests to the relevant business department for fulfilment. The customer first uploads a voice recording and this audio file is taken as input (provided it has the correct format) by the AI tool to determine the relevant service category it belong to. It does this using the pre-trained [XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-1b) model and [Sentence-transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeddings.

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


## References
[Huggingface](https://huggingface.co/)
[Sentence-transformers](https://pypi.org/project/sentence-transformers/)
[flask]()
