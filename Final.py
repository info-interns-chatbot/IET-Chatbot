import pandas as pd
import numpy as np
import csv
import datetime
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

df = pd.read_csv("train_data.csv")
df.dropna(inplace=True)

dr = pd.read_csv("Response.csv")

vectorizer = TfidfVectorizer()
vectorizer.fit(np.concatenate((df.Question, df.Answer)))

responser = TfidfVectorizer()
responser.fit(np.concatenate((dr.Response, dr.Meaning)))

Question_vectors = vectorizer.transform(df.Question)

response_vectors = responser.transform(dr.Response)

print("You can start chatting with me now.")
while True:
    input_question = input()
    flag = True

    input_question_vector = vectorizer.transform([input_question])

    similarities = cosine_similarity(input_question_vector, Question_vectors)
    
    cosine = np.delete(similarities, 0)
    max1 = cosine.max()
    print("rank : " + str(max1))

    closest = np.argmax(similarities, axis=1)
    
    print(closest)

    if(max1 >= 0.6):
        
        print("BOT: " + df.Answer.iloc[closest].values[0])
        if(input_question.lower() == "bye"):
            break
    elif(max1 <= 0.3):
        print("BOT: Please enter a valid question !!")
    else:
        print("BOT: Did you mean :")
        print("BOT: " + df.Question.iloc[closest].values[0])
        
        while(flag):
            input_response = input()
            input_response_vector = responser.transform([input_response])
            similar = cosine_similarity(input_response_vector, response_vectors)
            cos = np.delete(similar, 0)
            max2 = cos.max()
            print("rank of response : " + str(max2))
            if(max2 < 0.5):
                print("Invalid response")
            else:
                closest_response = np.argmax(similar, axis=1)
                print(dr.Meaning.iloc[closest_response].values[0])
                flag = False

                if(dr.Meaning.iloc[closest_response].values[0]=="yes"):
                    print("BOT: " + df.Answer.iloc[closest].values[0])
                    with open('train_data_test_1.csv', 'a', newline='') as train_data:
                        writer = csv.writer(train_data)
                        fields = [input_question, df.Answer.iloc[closest].values[0]]
                        writer.writerow(fields)
                else:
                    print("BOT: We will get back to you soon !!")
                    try:
                        with open('Unknown/Unknown_question_'+date+'.csv', 'a', newline='') as unknown_question:
                           writer = csv.writer(unknown_question)
                           fields = [input_question]
                           writer.writerow(fields)
                    except:
                        fileName = "Unknown/Unknown_question_"+date+"_.csv"
                        writer = csv.writer(fileName)
                        fields = [input_question]
                        writer.writerow(fields)
                    
                    
        
