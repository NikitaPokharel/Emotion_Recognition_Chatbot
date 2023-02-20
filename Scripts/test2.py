from keras.models import load_model
import tensorflow as tf
import speech_recognition as sr
from Scripts.get_emotion import *
from Scripts.generator import *

def predict(test,sentence,emotion,tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE):
    prediction = evaluate(test,sentence,emotion,tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE)
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size]
    )
    return predicted_sentence

def optimize():
    learning_rate = CustomSchedule(256)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    return optimizer

def get_mytext(audiofile):
    r = sr.Recognizer() 
    hellow=sr.AudioFile(audiofile)
    with hellow as source:
        audio = r.record(source)
    try:
        MyText2 = r.recognize_google(audio)
        print (MyText2)
    except sr.RequestError as e:
        MyText2="..."

    except sr.UnknownValueError:
        MyText2="..."       
    return MyText2

def generater(MyText,tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE,audiofile):
    if MyText=="...":
        reply='Are you trying to say something? I did not catch that. Could you please repeat?'
    else:
        #GETTING EMOTIION
        audio=audiofile
        MyText=MyText.lower()
        emotion=getemotion(audio,MyText)

        #GETTING REPLY
        optimizer=optimize()
        test=create_model(optimizer)
        test.load_weights('Scripts\\mymodel.h5')
        reply=predict(test,MyText,emotion,tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE)
    return reply
    

