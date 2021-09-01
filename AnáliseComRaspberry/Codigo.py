
 

import speech_recognition as sr
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from googletrans import Translator
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

# Obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

def mode1(text):
    positiveWords = np.array(pd.read_csv('negative-words_pt.csv',sep='\n'))
    negativeWords = np.array(pd.read_csv('positive-words_pt.csv',sep='\n'))

    stop_words = stopwords.words('portuguese')
    text = [x for x in text if x not in stop_words]
    text = text.apply(lambda x :x.astype(str).str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
    pos = 0
    neg = 0
    neu = 0
    for y in [x.lower().strip() for x in text.split(' ')]:
        if y in positiveWords:
            pos+=1
        elif y in negativeWords:
            neg+=1
        else:
            neu+=1
    graph(pos,neg,neu) 

def mode2(text):
    translator = Translator()
    text = translator.translate(text, dest='en')
    sid = SentimentIntensityAnalyzer()
   
    ss = sid.polarity_scores(text)
    graph(ss[3],ss[1],ss[2])
    
# Function that analyse text captured from the audio
def analyseText(text):
    mode1(text)
    mode2(text)

#Function that plot the graph, showing the positive and negative percetage
def graph(pos,neg,neu):
    sentimento = ['Neu','Neg','Pos']
    num = [neu,neg,pos]  
        
    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(12,5))

    plt.bar(sentimento,num, color = 'lightgrey')
    y = sorted(FreqDist(a).values(),reverse=True)
    for p in ax.patches:
            ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.35, p.get_height()))
    plt.rc("font",family="Times New Roman")
    ax.set_xlabel("Speech Content")
    ax.set_ylabel('Percentage of Speech Frequency')
    ax.set_title('Percentage of Speech Content')

# recognize speech using Google Cloud Speech
 
GOOGLE_CLOUD_SPEECH_CREDENTIALS = pd.read_json("credentials.json")
try:
    text = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
    print("Google Cloud Speech thinks you said " + text)
    analyseText(text)
except sr.UnknownValueError:
    print("Google Cloud Speech could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Cloud Speech service; {0}".format(e))

