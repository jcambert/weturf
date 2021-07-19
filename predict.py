import numpy as np
import pandas as pd
import re
import logging
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler,OneHotEncoder,LabelEncoder,PolynomialFeatures,FunctionTransformer

from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, make_column_transformer
NUMERICAL_FEATURES=['numPmu','rapport','age','nombreCourses','nombreVictoires','nombrePlaces','nombrePlacesSecond','nombrePlacesTroisieme','distance','handicapDistance','gain_carriere','gain_victoires','gain_places','gain_annee_en_cours','gain_annee_precedente']
CATEGORICAL_FEATURES=['sexe','hippo_code']
CALCULATED_FEATURES=['musique']
encoders={}
encoder=OneHotEncoder(handle_unknown = 'ignore')

music_pattern='([0-9,D,T,A,R][a,m,h,s,c,p,o]){1}'
music_prog = re.compile(music_pattern,flags=re.IGNORECASE)
music_penalities={'0':11,'D':6,'T':11,'A':11,'R':11}



def calculate_music(row,speciality='a'):
    try:
        music=row['musique']
        results = music_prog.findall(music)
        points=0
        for result in results:
            point= music_penalities[result[0]] if result[0] in music_penalities else int(result[0])
            points+=point
        return points/len(results) if len(results)>0 else 0
    except:
        # print("ERROR",music)
        pass
    finally:
        return 0
    # return points,points/len(results)


def encode(df,categories):
    for category in categories:
        if not category in encoders:
            encoders[category]=LabelEncoder()
            encoders[category].fit(df[category])
        df[category]= encoders[category].transform(df[category])

def col_place(row):
    return 1 if row['ordreArrivee'] in range(1,3) else 0

def load_train_file(filename):
    df=pd.read_csv(filename,sep=";")
    df['ordreArrivee'] = df['ordreArrivee'].fillna(0)
    places=df.apply (lambda row: col_place(row), axis=1)
    df['musique']=df.apply(lambda row:calculate_music(row),axis=1)  
    df=df[NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES]
    encode(df,CATEGORICAL_FEATURES)
    targets = places
    features = df
    return features,targets

def load_to_predict_file(filename):
    df=pd.read_csv(filename,sep=";")
    courses=df[['reunion','course']].drop_duplicates()
    encode(df,CATEGORICAL_FEATURES)
    return df,courses



def train(features,targets,test_size=0.2,random_state=0,shuffle=False):
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=test_size, random_state=random_state,shuffle=shuffle)
    # model = make_pipeline(StandardScaler(), SGDClassifier())
    # numerical_features = NUMERICAL_FEATURES
    # categorical_features = CATEGORICAL_FEATURES
    
    # encoder=OneHotEncoder()
    # print(features[categorical_features])
    # encoder.fit(features[categorical_features])
    # print(encoder.fit_transform(features[categorical_features]).toarray())
    # numerical_pipeline = make_pipeline(SimpleImputer(strategy='mean'), RobustScaler())
    # categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), encoder)
    # preprocessor = make_column_transformer((numerical_pipeline, numerical_features),(categorical_pipeline, categorical_features))
    # preprocessor = make_column_transformer((RobustScaler(),numerical_features),(OneHotEncoder(handle_unknown = 'ignore'),categorical_features),verbose=True)
    # model = make_pipeline(preprocessor, SGDClassifier(random_state=random_state))
    model = make_pipeline( PolynomialFeatures(),RobustScaler(),SGDClassifier(random_state=random_state))
    model.fit(features_train, targets_train)
    return model,features_train, features_test, targets_train, targets_test 

def predict_place(model,row):
    # row=row.reshape(1,row.shape[1])
    # print(type(row))
    x = np.asarray(row).reshape(1,len(row))
    
    numPmu=int(x[0,0])
    model.predict(x)
    result= model.predict(x)[0]==1
    # numPmu=row['numPmu']
    # result=model.predict(row)
    return numPmu,result

if __name__=='__main__':
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")
    
    training_files=['participants_trot_attele','participants_plat','participants_trot_monte']

    output={'date':[],'reunion':[],'course':[],'nom':[],'rapport':[],'numPmu':[],'state':[]}
    for training in training_files:
        try:
            features,targets=load_train_file(f"{training}.csv")
            to_predict,courses=load_to_predict_file(f"to_predict_{training}.csv")
            model,features_train, features_test, targets_train, targets_test =train(features,targets)
            print(model.score(features_test, targets_test))

            for course in courses.iterrows():
                x = np.asarray(course[1]).reshape(1,len(course[1]))
                r,c=x[0,0],x[0,1]
                participants=to_predict[(to_predict['reunion']==r) & (to_predict['course']==c)]
                details=participants[['date','nom']]
                participants['musique']=participants.apply(lambda row:calculate_music(row.musique),axis=1)  
                participants=participants[NUMERICAL_FEATURES+  CATEGORICAL_FEATURES+CALCULATED_FEATURES]
                # participants[CATEGORICAL_FEATURES]=encoder.transform(participants[CATEGORICAL_FEATURES])
                # participants=participants.drop('ordreArrivee',axis=1)
                nb_participants=participants.shape[0]
                # for participant in participants.iterrows():
                for i in range(nb_participants):
                    participant=participants.iloc[i]
                    detail=details.iloc[i]
                    numPmu,result=predict_place(model,participant)
                    if result:
                        output['date'].append(detail.date)
                        output['reunion'].append(r)
                        output['course'].append(c)
                        output['nom'].append(detail.nom)
                        output['rapport'].append(participant.rapport)
                        output['numPmu'].append(numPmu)
                        output['state'].append('place')
                        print(f"R{r}/C{c} -> {detail.nom}[{participant.rapport}] {numPmu} placé" )
        except Exception as ex:
            logging.warning(ex)
    pd.DataFrame.from_dict(output).to_csv('predicted.csv',header=True,sep=";",mode='w')