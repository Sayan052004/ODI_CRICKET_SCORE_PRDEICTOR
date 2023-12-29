import streamlit as st
import pandas as pd
import pickle
import xgboost
from xgboost import XGBRegressor
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = [
    'Australia',
    'India',
    'Bangladesh',
    'New Zealand',
    'South Africa',
    'England',
    'West Indies',
    'Afghanistan',
    'Pakistan',
    'Sri Lanka'
]

cities = ['Mirpur',
          'London',
          'Colombo',
          'Sydney',
          'Abu Dhabi',
          'Rangiri',
          'Melbourne',
          'Centurion',
          'Adelaide',
          'Perth',
          'Birmingham',
          'Dubai',
          'Auckland',
          'Johannesburg',
          'Wellington',
          'Brisbane',
          'Pallekele',
          'Cardiff',
          'Manchester',
          'Nottingham',
          'Southampton',
          'Durban',
          'Hamilton',
          'Sharjah',
          'Port Elizabeth',
          'Cape Town',
          'Christchurch',
          'Antigua',
          'Leeds',
          'Chandigarh',
          'Karachi',
          'Guyana',
          'Trinidad',
          'Napier',
          'St Lucia',
          'Hambantota',
          'Mumbai',
          'Jamaica',
          'St Kitts',
          'Chester-le-Street',
          'Barbados',
          'Hobart',
          'Lahore',
          'Delhi',
          'Ahmedabad',
          'Grenada',
          'Mount Maunganui',
          'Nagpur',
          'Visakhapatnam',
          'Jaipur',
          'Chennai',
          'Chittagong',
          'Harare',
          'Fatullah',
          'Bristol',
          'Nelson',
          'Dunedin',
          'Rajkot',
          'Kolkata',
          'Canberra',
          'Hyderabad',
          'Kanpur',
          'Cuttack',
          'Kuala Lumpur',
          'Dhaka',
          'Indore',
          'Bloemfontein',
          'Pune']

st.title('cricket_score_predictor')
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('select bowling team', sorted(teams))
citys = st.selectbox('select city', sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('current_score')
with col4:
    overs = st.number_input('overs done(works for over > 5)')
with col5:
    wickets = st.number_input('wickets out')

last_five = st.number_input('runs scored in last 5 overs')

if st.button('predict score'):
    ball_left = 300 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    input_df = pd.DataFrame(
        {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': citys,
         'current_score': [current_score],
         'ball_left': [ball_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]}
    )

    result = pipe.predict(input_df)
    st.text(result)
