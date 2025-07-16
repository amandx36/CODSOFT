from text_changer import text_transformer

import sys 
import pickle 
import string 
import nltk 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


vectorizer = pickle.load(open('vectorization.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


text_input = input("Enter the text to check , which genre it belongs to  >> ")
transfor_text = [text_transformer(text_input)]

vectorized_data = vectorizer.transform(transfor_text)
predict_data = model.predict(vectorized_data)





if predict_data[0] == 0 :
    print ("action")


elif predict_data[0] == 1 :
    print ("adult")

elif predict_data[0] == 2 :
    print ("adventure")

elif predict_data[0] == 3 :
    print ("animation")

elif predict_data[0] == 4 :
    print ("biography")


elif predict_data[0] == 5 :
    print ("comedy")

elif predict_data[0] == 6 :
    print ("crime")

elif predict_data[0] == 7 :
    print ("documentary")

elif predict_data[0] == 8 :
    print ("drama")


elif predict_data[0] == 9 :
    print ("family")

elif predict_data[0] == 10 :
    print ("fantasy")

elif predict_data[0] == 11 :
    print ("game-show ")

elif predict_data[0] == 12 :
    print ("history ")


elif predict_data[0] == 13 :
    print ("horror")

elif predict_data[0] == 14 :
    print ("music")

elif predict_data[0] == 15 :
    print ("Musical")

elif predict_data[0] == 16 :
    print ("mystery")


elif predict_data[0] == 17 :
    print ("news")

elif predict_data[0] == 18 :
    print ("reality-tv ")


elif predict_data[0] == 19 :
    print ("romance")

elif predict_data[0] == 20 :
    print ("sci-fi ")


elif predict_data[0] == 21 :
    print ("short")

elif predict_data[0] == 22 :
    print ("sport")

elif predict_data[0] == 23 :
    print ("talk-show")

elif predict_data[0] == 24 :
    print ("thriller ")


elif predict_data[0] == 25 :
    print ("war")

else :
    print ("western")
