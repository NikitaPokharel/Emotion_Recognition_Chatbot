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

def generater(tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE,audiofile):
    r = sr.Recognizer() 
    hellow=sr.AudioFile(audiofile)
    with hellow as source:
        audio = r.record(source)
    try:
                MyText2 = r.recognize_google(audio)
                MyText = MyText2.lower()

                print (MyText2)
                
                #GETTING EMOTIION
                audio=audiofile
                emotion=getemotion(audio,MyText)
                
                #GETTING REPLY
                optimizer=optimize()
                test=create_model(optimizer)
                test.load_weights('Scripts\\mymodel.h5')
                reply=predict(test,MyText,emotion,tokenizer, START_TOKEN,END_TOKEN,VOCAB_SIZE)
    except sr.RequestError as e:
        reply="Are you trying to say something? I did not catch that. Could you please repeat?"
        MyText2="..."
        # print("Could not request results; {0}".format(e)+'please speak again.')

    except sr.UnknownValueError:
        reply="Are you trying to say something? I did not catch that. Could you please repeat?"
        MyText2="..."
        # print("Unknown error occured,please speak again..")
    return reply,MyText2
    

