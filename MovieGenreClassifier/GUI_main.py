from text_changer import text_transformer 
import streamlit as st 
import pickle


import nltk


# this will find if not present than it will download the dependies 


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# now load the vectorizer model !!!! 

vectorizer = pickle.load(open('vectorization.pkl' , 'rb'))

# loading the model

model  = pickle.load(open('model.pkl','rb'))

# we are making the gui using the streamlit !! 

# now i making the GUI 

# this will make the heading of the web-page !!! 

st.set_page_config(page_title = "Movie Generic Predictor",page_icon= "🔮",layout = "centered")

# now we need to add the title of the page 

st.title("Movie Generic Predictor  !! ")

# add for the instruction for the users 

st.markdown ("Enter your text for the predictions ")

# changing the text area and adding the text input feature for taking the input from the user !!!!

text_input = st.text_area("Your message !! ", height = 150)

# now make the condition based statement for the predictions 


# creating the default dictionary for the storing all the default library !! that store the key and value pair !! 

mapping  = {
    0: "Action",
    1: "Adult",
    2: "Adventure",
    3: "Animation",
    4: "Biography",
    5: "Comedy",
    6: "Crime",
    7: "Documentary",
    8: "Drama",
    9: "Family",
    10: "Fantasy",
    11: "Game-Show",
    12: "History",
    13: "Horror",
    14: "Music",
    15: "Musical",
    16: "Mystery",
    17: "News",
    18: "Reality-TV",
    19: "Romance",
    20: "Sci-Fi",
    21: "Short",
    22: "Sport",
    23: "Talk-Show",
    24: "Thriller",
    25: "War"
}





if st.button('🛎️Predict IT') :
    if text_input.strip() == "":            # .strip()  := by default the strip function is used to remove the blank spaces in the text !!!  
        st.warning("You must have to enter the text . ")  # .strip(character)   :== this will remove the character from the string !! 
    
    else :
        # process the text which is taken input by the user !! 
        transformed_input = [text_transformer(text_input)]
        # i have converted into the list because the text data must be in the list for the vectorizer !!! 
        # now vectorized the text area 

        # now we have to vectorize the text area !! 
        # means converting the user_input_text_into__binary_so the computer can learn !! 
        
        transformed_input  =  vectorizer.transform(transformed_input)


        # now use the model for the prediction  according the user input data !!! 
        result  = model.predict(transformed_input)


        # now the value is stored into the result so that we can guess into the following category !! 
        predicted_value = mapping.get(result[0],"Describe more ")

        st.success(f"🎥 Predicted Genre is : {predicted_value}")
        # using the spilt function of streamlit for displaying !!! 

        

        
