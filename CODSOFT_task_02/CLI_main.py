

import pickle 
import pandas as pd 
import joblib 
import numpy as np




# loading the model !!! 
model = joblib.load("Model.pkl")
# print(type(model))


# print("job frequency")



job_frequency = pd.read_pickle("job_frequency.pkl")
# print(job_frequency)




# print(type(job_frequency))
# print(job_frequency.head(4))


# print("merchant data ")

merchant_repetation = pd.read_pickle("merchant_repetation.pkl")


# print("cities frequency")
cities_frequency = pd.read_pickle("cities_frequency.pkl")

# now its time for matching the input to the model input data brother !!! 

total_x_colums = pd.read_pickle("x_columns.pkl")

# now get the cities binary values !! 
# job_field = input("Enter the job title :-- ").strip()

# job_numeric_value = job_frequency.get(job_field,0)
# #print("the numeric value ",job_numeric_value)  # i used this to remove the extra space yarr 


# now make the cli !!! 

import pandas as pd

user_input = {
    
           "merchant" : input("Enter the merchant name:>").strip(),
           "amt" : float(input("Enter the Amount :>"))  ,   # now one important things i have to say strip function  did not work on the float data type !!!
           "Gender" : input("M/F :>").strip(),
           "city" : input("Enter the city :>").strip() ,
           "Zip" : int(input("Enter the zip code :>")),
           "Lat" : float(input("Enter the latitude :>")),
           "long" : float(input("Enter the longitude :)")),
           "city_pop" : int(input("Enter the city population :>")) ,   # also strip() function is not define for the int datatype 
           "job" : input("enter the job title :> ").strip(),
           "merch_lat" :float(input("Enter the merchant latitude :> ")),
           "merch_long" : float(input("Enter the longitude  :)"))  , 
           "category" : input("Enter the category :> ").strip(),
           "state" :input("Enter the state :) ").strip()
}

#        now i am making this into the dataframe brother !! 


# we are converting into the dataframe using the DataFrame function 
user_data = pd.DataFrame([user_input])



# now encoding the user data as our model is trained in that specific way brother !! 

user_data["Gender"] = user_data["Gender"].map({'M':0,'F':1})


# now encode / map the the the merchant with the column while the column in the training of the model !!  !! brother 

 



# 2️!!!!! .fillna(0)
# This handles unknown or unseen values.

# If .map() gives NaN (because the merchant is not in your training data) → .fillna(0) replaces that NaN with 0.
 

# convert the job frequency into the dictionary so  replace function will work 

job_frequency  = job_frequency.to_dict()


user_data["job"]  = user_data["job"].replace(job_frequency).fillna(0)

user_data["job"] = pd.to_numeric(user_data["job"], errors='coerce').fillna(0)

# now map the city with the column form in during the training of the model if anything doesnot match with trained data than dont raise error just give him the default value 0 with the help of .fillna(0) inbuilt python function !! 



cities_frequency  =   cities_frequency.to_dict()
user_data["city"] = user_data["city"].map(cities_frequency).fillna(0) 


# now mapping the merchant !! 


merchant_repetation = merchant_repetation.to_dict()
user_data["merchant"] = user_data["merchant"].map(merchant_repetation).fillna(0) 


# now we are doing the one hot encoding for the category and states !!! 

user_data = pd.get_dummies(user_data , columns= ['category','state'])


# thus we have to give the dummy data to fill the need of model as they were trained in such an order !!!! 

print("Type of total_x_colums elements: ", type(total_x_colums[0]))


for col in total_x_colums :
    if col not in user_data.columns :
        user_data[col] = 0    # add the missing the column with the zeros !!! 



# arranging the column in such a way model is trained 

user_data = user_data[total_x_colums]



# make the prediction for the user yarr 

prediction  = model.predict(user_data)
if prediction[0] == 1 :
    print("Alert  fraud ")

else :
    print("Not a fraud !!!")




