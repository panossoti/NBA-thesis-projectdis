# Importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.max_columns = 99
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df_players= pd.read_csv("/Users/panagiotissotirellos/Desktop/Dis_csv_data/player.csv")
df_frame= pd.read_csv("/Users/panagiotissotirellos/Desktop/Dis_csv_data/dimitri.csv")
df_frame.dropna(inplace= True,how="all")
df_frame.drop(["Team Impact weighted PER","Opposition impact Weighted PER","Home Team Winstreak","Away Team Winstreak"], axis=1 ,inplace=True)

# Calculating Feature weighted average
def weighted_average(x):
    total_mp = sum(x["MP"])
    pers=0
    for i,r in x.iterrows():
        pers += r["MP"]/total_mp*r["PER"]
    return pers

all_stars=["Stephen Curry","Kevin Durant","Russell Westbrook","LeBron James","Kawhi Leonard","James Harden","Anthony Davis","DeMarcus Cousins","LaMarcus Aldridge","Kyle Lowry","Pau Gasol","DeMar DeRozan","Isaiah Thomas","Jimmy Butler","Paul Millsap","Paul George","Kemba Walker","DeAndre Jordan","Kyrie Irving","John Wall","Draymond Green","Kevin Love","Gordon Hayward","Klay Thompson","Marc Gasol"]

# all stars and non all stars PER
teams_non_star_per = {}
teams_star_per = {}
for team, team_df in df_players.groupby(["Tm"]):
    non_star_df = team_df[~df_players.Player.isin(all_stars)]
    avg_per_non_star = weighted_average(non_star_df)
    teams_non_star_per[team] = avg_per_non_star


    star_df = team_df[df_players.Player.isin(all_stars)]
    avg_per_star = weighted_average(star_df)
    teams_star_per[team] = avg_per_star
    if (teams_star_per[team] == 0) :
        teams_star_per[team] = teams_non_star_per[team]
print(df_frame.shape)
# add weighted team PER column without an all star to the data set
df_frame["Weighted_Team_PER"]=0
for i,r in df_frame.iterrows():
     df_frame.loc[i,"Weighted_Team_PER"]= teams_non_star_per[r["Home Team 1"]]
# add weighted team PER column with the all stars to the data set
df_frame["Weighted_opposition_PER"]=0
for i,r in df_frame.iterrows():
    df_frame.loc[i,"Weighted_opposition_PER"]= teams_star_per [r["Away Team 2"]]
print(df_frame.head())

# create win streak column in the dataframe
df_frame["win_streak_Home"] = df_frame["Streak Home Team"].apply(lambda x: int(x[1:]) if x[0]=="W" else 0 )

df_frame["lose_streak_Home"] = df_frame["Streak Home Team"].apply(lambda x: int(x[1:]) if x[0]=="L" else 0 )

df_frame["win_streak_Away"] = df_frame["Streak Away Team"].apply(lambda x: int(x[1:]) if x[0]=="W" else 0 )

df_frame["lose_streak_Away"] = df_frame["Streak Away Team"].apply(lambda x: int(x[1:]) if x[0]=="L" else 0 )

print(df_frame.head())
# Define X and Y variables to split train and testing Dataset
X=df_frame[["Did Star Player  play" , "Weighted_Team_PER" , "Weighted_opposition_PER" , "win_streak_Home" , "win_streak_Away" ]]
Y=df_frame[["Home Team Won" ]]


# Training of the Model
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=2)

lr = LogisticRegression(C=1e15)
lr.fit(X_train,Y_train)

y_predict= lr.predict(X_test)

lr.score(X_test,Y_test)



# Model Accuracy
model_accuracy=accuracy_score(y_predict,Y_test,normalize= True,)
print("Model Accuracy in predicting 20% of Regular Season Games: ",model_accuracy,"%")
print(Y_test["Home Team Won"].value_counts())
# Null Accuracy of my model meaning This means that a dumb model that always predicts 0 would be right 68% of the time
# It's a good way to know the minimum we should achieve with our models
print("Null Accuracy: " , 1-Y_test["Home Team Won"].mean())

# Random forest Regression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50,random_state=42)
rf.fit(X_train,Y_train)
y_predict_rf = rf.predict(X_test)
rf.score(X_test,Y_test)
rf_accuracy =  accuracy_score(y_predict_rf,Y_test)
print("Random Forest Accuracy score: ",rf_accuracy," % ")

import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_predict)
# Visualise ("Confusion Matrix")
print(cm)
plt.figure(figsize=(6,4))
plt.xlabel("predicted")
plt.ylabel("Actual")
labels = ["True Neg’,’False Pos’,’False Neg’,’True Pos"]
categories = ["Zero", "One"]
group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt="", cmap='Blues')
plt.draw()
# get importance
importance = lr.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

# Logistic Regression Feauture importances
feature_importances = pd.DataFrame(rf.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)

report = classification_report(Y_test,y_predict)
print(report)

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier (max_depth= 3,random_state=0)
clf.fit(X_train, Y_train)

# Visualise Decision Tree
fn= ["Did Star Player  play" , "Weighted_Team_PER" , "Weighted_opposition_PER" , "win_streak_Home" , "win_streak_Away" ]
cn= [" Home Team Win", " Home Team Loss"]
fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=300)
tree.plot_tree(clf,feature_names = fn , class_names=cn,filled = True)
fig.savefig('random_forest_tree.png')
plt.show()

# Feature importance visualisation
plt.style.use('seaborn-dark')
plt.rcParams['axes.facecolor'] = 'black'
features = ['Weighted Op PER (0.348)', 'Weighted Team PER','Star Player','Win Streak H','Win Streak A']
weight = [0.348,0.301,0.148,0.103,0.098]
plt.title('Feauture Importances')
plt.xlabel('Features')
plt.ylabel('Weight')
plt.bar(features,weight,color=['yellow', 'red', 'green', 'blue', 'cyan'])
plt.show()

# Pie chart
labels_=['Weighted Opposition PER', 'Weighted Team PER','Star Player','Win Streak Home','Win Streak Away']
size= [0.348,0.301,0.148,0.103,0.098]
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99',"yellow"]
# explosion
explode = (0.05, 0.05, 0.05, 0.05)
# title
print()
plt.xlabel("Feauture importances")
print()
#draw circle
plt.pie(size, colors=colors, labels=labels_, autopct='%1.1f%%', startangle=90, pctdistance=0.85,shadow= True)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle

plt.tight_layout()
plt.show()









