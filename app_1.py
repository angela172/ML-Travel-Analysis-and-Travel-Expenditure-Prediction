import streamlit as st
import pandas as pd
from PIL import Image  
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import xgboost as xg
from plotly_calplot import calplot
import matplotlib.pyplot as plt
import time
from streamlit_option_menu import option_menu
from sklearn.metrics import r2_score
import plotly.io as pio
import plotly.express as px
import joblib
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def sidebar_bg(side_bg):
    side_bg_ext = 'jpg'

    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{
            background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
            color: white;
        }}
        .sidebar-text {{
            color: white; /* Change 'blue' to your desired text color */
        }}
    </style>
""",
        unsafe_allow_html=True,
    )

side_bg = r'C:\Users\11 PrO\Desktop\ML project papers\travel10.jpg'

sidebar_bg(side_bg)


st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 250px;
        max-width: 500px;
        color: white;
    }   
    """,
    unsafe_allow_html=True,
)   

st.sidebar.title("Travel analysis+accommodation cost prediction")
st.sidebar.image("https://media.giphy.com/media/KenFWffodaDVaceDa6/giphy.gif",width = 100)
st.sidebar.markdown("")
side_bar = st.sidebar.radio('What would you like to view?', [ 'About the dataset', 'Data Analysis🔎', 'Accommadation model analysis','Predict Accommodation cost','About the model'])

if side_bar == 'About the dataset':
    header = st.container()
    features = st.container()

    with header:
        
        text_col,image_col = st.columns((5.5,1))
        with text_col:
            st.title("Understanding the Dataset")
            st.markdown("The dataset contains information of the traveler and travel information including the travel expenses")
            st.write("")
            
            markdown_text = '''
            ```
 
                    Data columns (total 13 columns):
                    #   Column                 Non-Null Count  Dtype 
                    ---  ------                 --------------  ----- 
                    0   Trip ID                137 non-null    int64 
                    1   Destination            137 non-null    object
                    2   End date               137 non-null    object
                    3   Duration (days)        137 non-null    int64 
                    4   Traveler name          137 non-null    object
                    5   Traveler age           137 non-null    int64 
                    6   Traveler gender        137 non-null    object
                    7   Traveler nationality   137 non-null    object
                    8   Accommodation type     137 non-null    object
                    9   Accommodation cost     137 non-null    int64 
                    10  Transportation type    137 non-null    object
                    11  Transportation cost    137 non-null    int64 
                    12  Overall Travel Budget  137 non-null    int64 
            ```
            '''
            st.markdown(markdown_text)
            st.write("")    

#---------------------------------SIDE BAR OPTIONS---------------------------------------------
elif side_bar == 'Data Analysis🔎':  
    text,img2 = st.columns((2,1))
    with text:
        st.title("Analysis of the dataset")
        st.write("")
        
    with img2:
        st.image("https://media.giphy.com/media/l46Cy1rHbQ92uuLXa/giphy.gif", width = 150)
        
    st.write("----")
    st.write("select the options below to discover insights in the dataset")
    options2 = ["Display Top Destinations", "What are the 5 most common transportation types","Display 5 Most Common Accommodation Type","Which gender travels the most?","Nationalities that mostly travel"]
    with st.form(key='Travel_form'):
        selected_option2 = st.selectbox("Select an insight: ", options2)
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        markdown_text = f" Insight selected: `{selected_option2}\n`"
        st.sidebar.markdown(markdown_text)
        if selected_option2 == "Display Top Destinations":
            st.write("Here's a plot that showcases the most popular destinations")
            i1 = Image.open(r"C:\Users\11 PrO\Desktop\ML project papers\populardest.png")
            st.image(i1, width = 500)

        elif selected_option2 == "What are the 5 most common transportation types": 
            st.write("Here's a plot that showcases the most common transportation types")
            i2 = Image.open(r"C:\Users\11 PrO\Desktop\ML project papers\transportation types.png")
            st.image(i2, width = 500)

        elif selected_option2 == "Display 5 Most Common Accommodation Type": 
            st.write("Here's a plot of the 5 most common Accomadation types")
            i3 = Image.open(r"C:\Users\11 PrO\Desktop\ML project papers\accomd.png")
            st.image(i3, width = 500)

        elif selected_option2 == "Which gender travels the most?": 
            st.write("This plot showcases which gender travels the most")
            i4 = Image.open(r"C:\Users\11 PrO\Desktop\ML project papers\travelgender.png")
            st.image(i4, width = 500)

        elif selected_option2 == "Nationalities that mostly travel": 
            st.write("The plot showcases which all type of nationalities have travelled and also which is the nationality that has travelled the most")
            i5 = Image.open(r"C:\Users\11 PrO\Desktop\ML project papers\nationality.png")
            st.image(i5, width = 500)

elif side_bar == 'Accommadation model analysis':  
    text,img2 = st.columns((2,1))
    with text:
        st.title("Accommodation cost analysis")
        st.write("")
    with img2:
        st.image("https://media.giphy.com/media/JrXas5ecb4FkwbFpIE/giphy.gif", width = 150)
    st.write("Model: Random Forest Regressor")
    st.write("Here's a plot that showcases the predicted Y values and the actual Y values for Random Forest Regression")
    st.write("R^2: 0.8556834509575529")
    im2 = Image.open(r"C:\Users\11 PrO\Downloads\new ml\down1.png")
    st.image(im2, width = 500)
           
          
elif side_bar == 'Predict Accommodation cost':  
    df = pd.read_csv(r"C:\Users\11 PrO\Desktop\Travel details dataset2.csv")
    X = df.iloc[:, [1, 3,8,12]].values
    y= df.iloc[:, 9].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X[:,0] = le.fit_transform(X[:,0])
    X[:,2] = le.fit_transform(X[:,2])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)
    from sklearn.ensemble import RandomForestRegressor
    tree = RandomForestRegressor()
    tree.fit(X, y) 
    text,img2 = st.columns((2,1))
    with text:
        st.title("Accommodation cost predictor")
        st.write("")   
    with img2:
        st.image("https://media.giphy.com/media/6NG1WV8bEChNJH3Z9R/giphy.gif", width = 150)
    st.write("Model used: Random Forest Regressor")  
    with st.form(key='transcost_form'):
            Destination = st.text_input("Enter Destination:")
            Duration = st.number_input("Enter the duration:")
            Travel_budget = st.number_input("Enter your overall Travel Budget( in $):")
            opt1=['Hotel', 'Resort', 'Villa', 'Airbnb', 'Hostel', 'Raid','Vacation rental', 'Guesthouse']
            Accomod = st.selectbox("Select a Model: ", opt1)
            submit_button = st.form_submit_button(label='Submit')
            if submit_button:
                new_data = [Destination,Duration, Accomod, Travel_budget]
                new_data[0] = le.fit_transform([new_data[0]])[0]
                new_data[2] = le.fit_transform([new_data[2]])[0]
                new_data_reshaped = [new_data]
                predict1 = tree.predict(new_data_reshaped)
                st.write("")
                st.write("Your Accommodation cost: dhs", predict1[0])
                remain=Travel_budget-predict1[0]
                st.write("Remaining buget: dhs",remain)

elif side_bar == "About the model":
    st.title("About the model")
    markdown2='''
    The model built in this project helps us to not only gain insights of the dataset used but also helps us to predict the travel expenditures such as
    Accommadation cost  when trip planning based on the overall travel budget of the indiviual. The dataset for this project has been taken from Kaggle and has been modified
    for preproccessing and standaradization purposes.
    
    The required features that have been selected from the dataset for these models are: 
    - Destination: The name of the city or country visited by the traveler.
    - Duration (days): The number of days the traveler spent on the trip.
    - Accommodation type: The type of accommodation the traveler stayed in, such as hotel, hostel, or Airbnb.
    - Accommodation cost: The cost of the accommodation for the entire trip
    - Transportation type: The mode of transportation used by the traveler, such as plane, train, or car.
    - Transportation cost: The cost of transportation for the entire trip.
    - Overall Travel Budget: The Overall Travel Budget of the Traveler.
    
    Moreover, the features from the original dataset have also been used to build a `correlation matrix` which shows the dependancies or the correlations 
    of different features. 
    
    '''
    st.markdown(markdown2)
    img6 = Image.open(r"C:\Users\11 PrO\Downloads\new ml\corr.png")
    st.image(img6, caption=" ", use_column_width=True)
    st.write("")
    
    markdown3= '''
    Moving on, The technique used for the purpose of this project is regression. Out of the many regression models, The Random Forest Regressor
    tends to give a good R-square value for predicting the Accommodation cost.
    '''
    st.markdown(markdown3)
    st.write("")