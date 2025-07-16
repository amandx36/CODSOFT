import nltk 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer



import sys 
import pickle 





import string 
ps = PorterStemmer()


#downloading the important libraries ! 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')




def text_transformer(text):

     # a ) lower case conversion !!    
    text  = text.lower()


    #  b ) converting the sentence into the list of words 
    text = nltk.word_tokenize(text)  #   (eg Aman deep loves oranges )  output :--> ['aman', 'deep', 'loves', 'oranges'] 



    #  c ) now removing the special character using the .isalnum() and 

    #store in the list after that you have to pass the list into the text 


#    I/p :== "Aman deep!!!@  !!!a@@ loves oranges"   
    l = []
    for i in text :
        if (i.isalnum()):
            l.append(i)

    text = l[:]
   

    # O/p :== ['aman', 'deep', 'a', 'loves', 'oranges']

    l.clear()





# d ) removing the puncutation and the stop words !! using the[[   string.punctuation and the stopwords.words('english')   ]] 

    for i in text:
        if i not in string.punctuation and i not in stopwords.words('english'):
            l.append(i)


    text = l[:]

    l.clear()

    
# 3 ) convert the following  words in  its root form !! 
    for i in text:
        l.append(ps.stem(i))


    text = l[:] 

 
    l.clear()

    # now the output is ['aman', 'deep', 'a', 'love', 'orang']

    
     # now the output will be [  ['aman', 'deep', 'love', 'orang'] ]
    
#    now remove the list and turn into the normal text yarrr !! using the   " ".join(argument)

    return " ".join(text)
    # o/p will be :==   aman deep love orang





import nltk
print("NLTK version:", nltk.__version__)



