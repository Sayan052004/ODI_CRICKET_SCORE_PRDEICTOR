
import pyttsx3  # install
import datetime
import speech_recognition as sr  # install
import pyaudio # install
import wikipedia # install
import webbrowser # install
import os
import openai
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # inbuild system voice Male / Female (i used Male voice )

def speak(audio):
    engine.say(audio)
    engine.runAndWait()  # Method

def wishme():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("Good morning, sir")
    elif 12 <= hour < 18:
        speak("Good afternoon, sir")
    else:
        speak("Good evening, sir")
    speak(' i am Jarvis here. Tell me how may I assist you.')

def takeinput():
    r = sr.Recognizer()
    with sr.Microphone() as source:  # This a class of python
        print('Listening.......')
        r.pause_threshold = 1 # generate the minimum sound capacity 
        audio = r.listen(source, timeout=60)
    try:
        print('Recognizing.....')
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")
    except Exception as e:
        print('Say that again, please......')
        return "None"
    return query

if __name__ == "__main__":
    wishme()
    while True:
        query=takeinput().lower()

        if 'wikipedia' in query:
            print("Searching wikipedia......")
            query=query.replace('wikipedia','')
            result=wikipedia.summary(query,sentences = 5)
            speak("According to wikipedia")
            print(result)
            speak(result)
        elif 'open youtube' in query:
            webbrowser.open('youtube.com')
        elif 'open google' in query:
            webbrowser.open('google.com')
        # elif 'open  stackoverflow' in query:
        #     webbrowser.open('stackoverflow.com')
        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            print(strTime)
            speak(strTime)
        elif 'open code' in query:
            codePath= "C:\\Users\\sayan\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"     # \\ is used for escape the character
            os.startfile(codePath)
        elif 'open file' in query:
            filePath= "C:\\Users\\sayan\\OneDrive\\Documents\\ODI_matchs\\data"
            os.startfile(filePath)
        # elif 'open python' in query:
        #     python="C:\\Program Files\\JetBrains\\PyCharm Community Edition 2023.2.3\bin\\pycharm64.exe"
            # os.startfile(python)
        elif 'open stores' in query:
            playstores="C:\\Program Files\\Google\\Play Games\\Bootstrapper.exe"
            os.startfile(playstores)
        elif 'open wikipedia' in query:
            path="https://www.wikipedia.org/"
            os.system(path)
        elif 'open microsoft word' in query:
            wordpath="C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE"
            os.startfile(wordpath)
        elif 'open microsoft excel' in query:
            excelpath="C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.EXE"
            os.startfile(excelpath)
        elif 'open powerpoint' in query:
            powerpointpath="C:\\Program Files\\Microsoft Office\\root\\Office16\\POWERPNT.EXE"
            os.startfile(powerpointpath)

