import streamlit as st
import pickle as pkl
import pandas as pd
teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

model = pkl.load(open("pipe.pkl", "rb"))
st.title('IPL Winner Predictor')

cols1, cols2 = st.columns(2)
with cols1:
    batting_team = st.selectbox("Select the batting team", sorted(teams))

with cols2:
    bowling_team = st.selectbox("Select the bowling team", sorted(teams))

selected_city = st.selectbox("Select the hosting city", sorted(cities))

target = st.number_input("Target")

cols3, cols4, cols5 = st.columns(3)

with cols3:
    scores = st.number_input("Score")

with cols4:
    overs = st.number_input("Overs Completed")

with cols5:
    wickets = st.number_input("Wickets Out")

if st.button("Predict Probabilities"):
    runs_left = target - scores
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = scores/overs
    rrr = (runs_left*6)/balls_left

    input_data = pd.DataFrame({'batting_team':[batting_team], 'bowling_team':[bowling_team],
    'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],
    'wickets':[wickets],'total_runs_x':[target], 'crr':[crr], 'rrr':[rrr]})

    result = model.predict_proba(input_data)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + ":- " + str(round(win*100)) + "%")
    st.header(bowling_team + ":- " + str(round(loss*100)) + "%")