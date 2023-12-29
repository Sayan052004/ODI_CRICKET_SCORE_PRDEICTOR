import numpy as np
import pandas as pd
from yaml import safe_load
import os 
from tqdm import tqdm

filenames=[]
for file in os.listdir('odi'):
    filenames.append(os.path.join('odi',file))

filenames[0:5]

final_df=pd.DataFrame()
counter=1
for file in tqdm(filenames):
    with open(file,'r') as f:
        df=pd.json_normalize(safe_load(f))
        df['match id']=counter
        final_df=final_df.append(df)
        counter+=1
final_df

backup=final_df.copy()
final_df

final_df.drop(columns=[
    'meta.data_version',
    'meta.created',
    'meta.revision',
    'info.supersubs.Asia XI',
    'info.supersubs.Zimbabwe',
    'info.supersubs.Sri Lanka',
    'info.supersubs.Bangladesh',
    'info.supersubs.New Zealand',
    'info.supersubs.India',
    'info.supersubs.ICC World XI',
    'info.supersubs.South Africa',
    'info.supersubs.Pakistan',
    'info.supersubs.West Indies',
    'info.supersubs.Australia'
    
],inplace=True)

final_df['info.gender'].value_counts()

final_df=final_df[final_df['info.gender']=='male']
final_df.drop(columns=['info.gender'],inplace=True)
final_df

final_df['info.match_type'].value_counts()
final_df['info.overs'].value_counts()
import pickle
pickle.dump(final_df,open('data_sets.pkl','wb'))
matches=pickle.load(open('data_sets.pkl','rb'))
matches
count = 1
delivery_df = pd.DataFrame()
for index, row in matches.iterrows():
    if count in [75,108,150,180,268,360,443,458,584,748,982,1052,1111,1226,1345]:
        count+=1
        continue
    count+=1
    ball_of_match = []
    batsman = []
    bowler = []
    runs = []
    player_of_dismissed = []
    teams = []
    batting_team = []
    match_id = []
    city = []
    venue = []
    for ball in row['innings'][0]['1st innings']['deliveries']:
        for key in ball.keys():
            match_id.append(count)
            batting_team.append(row['innings'][0]['1st innings']['team'])
            teams.append(row['info.teams'])
            ball_of_match.append(key)
            batsman.append(ball[key]['batsman'])
            bowler.append(ball[key]['bowler'])
            runs.append(ball[key]['runs']['total'])
            city.append(row['info.city'])
            venue.append(row['info.venue'])
            try:
                player_of_dismissed.append(ball[key]['wicket']['player_out'])
            except:
                player_of_dismissed.append('0')
    loop_df = pd.DataFrame({
            'match_id':match_id,
            'teams':teams,
            'batting_team':batting_team,
            'ball':ball_of_match,
            'batsman':batsman,
            'bowler':bowler,
            'runs':runs,
            'player_dismissed':player_of_dismissed,
            'city':city,
            'venue':venue
        })
    delivery_df = delivery_df.append(loop_df)

delivery_df

def bowl(row):
    for team in row['teams']:
        if team != row['batting_team']:
            return team

delivery_df['bowling_team'] = delivery_df.apply(bowl,axis=1)
delivery_df

delivery_df.drop(columns=['teams'],inplace=True)
delivery_df['batting_team'].unique()

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

delivery_df = delivery_df[delivery_df['batting_team'].isin(teams)]
delivery_df = delivery_df[delivery_df['bowling_team'].isin(teams)]

delivery_df

output = delivery_df[['match_id','batting_team','bowling_team','ball','runs','player_dismissed','city','venue']]
output

pickle.dump(output,open('dataset_level2.pkl','wb'))
import pandas as pd
import pickle
import numpy as np
df=pickle.load(open('dataset_level2.pkl','rb'))
df
df.isnull().sum()
df[df['city'].isnull()]['venue'].value_counts()

cities=np.where(df['city'].isnull(),df['venue'].str.split().apply(lambda x:x[0]),df['city'])
df['city']=cities
df.isnull().sum()
df.drop(columns=['venue'],inplace=True)
df

eligible_citys=df['city'].value_counts()[df['city'].value_counts()>1500].index.tolist()
df=df[df['city'].isin(eligible_citys)]
df

df['current_score']=df.groupby('match_id').cumsum()['runs'] # cumulative_sum() ----> cumusum()
df

df['over']=df['ball'].apply(lambda x:str(x).split(".")[0])
df['ball_no']=df['ball'].apply(lambda x:str(x).split(".")[1])
df
df['ball_bowled']=(df['over'].astype('int')*6)+df['ball_no'].astype('int')
df

df['ball_left']=300-df['ball_bowled']
df['ball_left']=df['ball_left'].apply(lambda x:0 if x<0 else x)
df

df['player_dismissed']=df['player_dismissed'].apply(lambda x:0 if x=='0' else 1)
df['player_dismissed']=df['player_dismissed'].astype('int')
df['player_dismissed']=df.groupby('match_id').cumsum()['player_dismissed']
df['wickets_left']=10-df['player_dismissed']
df

df['crr']=(df['current_score']*6)/df['ball_bowled']
df

groups=df.groupby('match_id')
match_ids=df['match_id'].unique()
last_five=[]
for id in match_ids:
    last_five.extend(groups.get_group(id).rolling(window=30).sum()['runs'].values.tolist()) # pandas rolling()
df['last_five']=last_five
df

final_df=df.groupby('match_id').sum()['runs'].reset_index().merge(df,on='match_id')
final_df=final_df[['batting_team','bowling_team','city','current_score','ball_left','wickets_left','crr','last_five','runs_x']]
final_df

final_df.isnull().sum()
final_df.dropna(inplace=True)
final_df.isnull().sum()
final_df=final_df.sample(final_df.shape[0])
final_df

x=final_df.drop(columns=['runs_x'])
y=final_df['runs_x']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

x_train

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error

trf=ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')

pipe=Pipeline(steps=[
    ('step1',trf),
    ('step2',StandardScaler()),
    ('step3',XGBRegressor(n_estimators=1000,learning_rate=0.2,max_depth=12,random_state=1))
])

pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))

pickle.dump(pipe,open('pipe.pkl','wb'))
df=pickle.load(open('pipe.pkl','rb'))
df

eligible_citys

import xgboost
xgboost.__version__
