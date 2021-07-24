from operator import indexOf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import logging
from sklearn.base import TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler,OneHotEncoder,LabelEncoder,PolynomialFeatures,FunctionTransformer
from sklearn.model_selection import cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.feature_selection import RFE,RFECV,VarianceThreshold
from sklearn.model_selection import  learning_curve
from sklearn.metrics import confusion_matrix
HEADER_COLUMNS=['date','reunion','course','nom']
NUMERICAL_FEATURES=['numPmu','rapport','age','nombreCourses','nombreVictoires','nombrePlaces','nombrePlacesSecond','nombrePlacesTroisieme','distance','handicapDistance','gain_carriere','gain_victoires','gain_places','gain_annee_en_cours','gain_annee_precedente','sexe','musique']
CATEGORICAL_FEATURES=['hippo_code','deferre']
CALCULATED_FEATURES=[]
encoders={}
encoder=OneHotEncoder(handle_unknown = 'ignore')

music_pattern='([0-9,D,T,A,R][a,m,h,s,c,p,o]){1}'
music_prog = re.compile(music_pattern,flags=re.IGNORECASE)
music_penalities={'0':11,'D':6,'T':11,'A':11,'R':11}

DEFAULT_MUSIC=11

class FuncTransformer(TransformerMixin):
    def __init__(self, func):
        self.func = func
    def fit(self,X,y=None, **fit_params):
        return self
    def transform(self,X, **transform_params):
        return self.func(X)
        
def process_music_transform(df):
    if "musique" in df:
        df_=df["musique"].map(lambda r:calculate_music(r))
        df.drop(['musique'],axis=1)
        
        return pd.concat([df,df_], ignore_index=True)
    return df

def calculate_music(music,speciality='a'):
    points=0
    results=[]
    try:
        # music=row['musique']
        results = music_prog.findall(music)
        for result in results:
            point= music_penalities[result[0]] if result[0] in music_penalities else int(result[0])
            points+=point
    except:
        pass
    finally:
        return points/len(results) if len(results)>0 else DEFAULT_MUSIC
        
SEXES= ['MALES','FEMELLES','HONGRES']
def sexe_converter(sexe):
    if not sexe in SEXES:
        return -1
    return indexOf(SEXES,sexe)
def encode(df,categories):
    for category in categories:
        if not category in encoders:
            encoders[category]=LabelEncoder()
            encoders[category].fit(df[category])
        df[category]= encoders[category].transform(df[category])

def place_converter(row):
    return 1 if row['ordreArrivee'] in range(1,3) else 0

def load_file(filename,is_predict=False):
    
    df=pd.read_csv(filename,sep=";",header=0,usecols=HEADER_COLUMNS+NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES+['ordreArrivee'],dtype={'numPmu':np.number},low_memory=False,converters={'musique':calculate_music,'sexe':sexe_converter})
    # for c in df.columns:
    #     print(c)
    # print(df.head())
    if not is_predict:
        df['ordreArrivee'] = df['ordreArrivee'].fillna(0)
        places=df.apply (lambda row: place_converter(row), axis=1)
        # df['musique']=df.apply(lambda row:calculate_music(row),axis=1)  
        # df=df[NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES]
        # encode(df,CATEGORICAL_FEATURES)
        targets = places
        features = df[NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES]
        return features,targets
    else:
        courses=df[['reunion','course']].drop_duplicates()
        participants=df[['date','nom','numPmu']]
        return df[HEADER_COLUMNS+NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES],courses,participants

def load_to_predict_file(filename):
    df=pd.read_csv(filename,sep=";")
    courses=df[['reunion','course']].drop_duplicates()
    # df=df[NUMERICAL_FEATURES+CATEGORICAL_FEATURES+CALCULATED_FEATURES]
    # encode(df,CATEGORICAL_FEATURES)
    return df,courses

def learning_curve_data(model,X_train,y_train,train_sizes=None,cv=None):
    if not train_sizes:
        train_sizes=np.linspace(0.2,1.0,5)
    N,train_score,val_score=learning_curve(model,X_train,y_train,train_sizes=train_sizes,cv=cv)
    return N,train_score,val_score
def train(features,targets,test_size=0.3,random_state=5,shuffle=False):
    
    # GridSearchCv Result
    #{'sgdclassifier__eta0': 0.05, 'sgdclassifier__learning_rate': 'optimal', 'sgdclassifier__loss': 'squared_hinge', 'sgdclassifier__max_iter': 5000, 'sgdclassifier__n_jobs': 1, 'sgdclassifier__shuffle': True}
    classifier=SGDClassifier(random_state=random_state,loss='squared_hinge',shuffle=True,learning_rate='optimal')
    # classifier=SGDClassifier(random_state=random_state)
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=test_size, random_state=random_state,shuffle=shuffle)

    numerical_pipeline=make_pipeline(SimpleImputer(fill_value=0), RobustScaler())
    categorical_pipeline=(make_pipeline(OneHotEncoder(handle_unknown = 'ignore')))
    preprocessor=make_column_transformer(
        (numerical_pipeline,NUMERICAL_FEATURES),
        (categorical_pipeline,CATEGORICAL_FEATURES))
    model_=make_pipeline(preprocessor,PolynomialFeatures(),VarianceThreshold(0.1),classifier)
    
    #  "LEARNING CURVE"
    # N,train_score,val_score=learning_curve(model_,features_train,targets_train,train_sizes=np.linspace(0.1,1.0,10),cv=5)
    # print(N)
    # print(train_score.mean(axis=1))
    # print( val_score.mean(axis=1))

    # plt.plot(N, train_score.mean(axis=1), label='train')
    # plt.plot(N, val_score.mean(axis=1), label='validation')
    # plt.xlabel('train_sizes')
    # plt.legend()

    #"GridSearchCV Test"
    # param_grid={'sgdclassifier__loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive'],
    #             'sgdclassifier__max_iter':np.arange(10,1000,2),
    #             'sgdclassifier__shuffle':[True,False],
    #             'sgdclassifier__n_jobs':[1,2,3,4],
    #             'sgdclassifier__learning_rate':['constant','optimal','invscaling','adaptive'],
    #             'sgdclassifier__eta0':[0.05],
    #             'sgdclassifier__max_iter':[5000]}
    # cv=StratifiedKFold(4)
    # grid=GridSearchCV(model_,param_grid,cv=cv)
    # grid.fit(features_train,targets_train)
    # model_=grid.best_estimator_
    # print(grid.best_score_,grid.best_params_)
    
    model_.fit(features_train,targets_train)

    return model_,features_train, features_test, targets_train, targets_test 

def predict_place(model,row):
    # x = np.asarray(row).reshape(1,len(row))
    numPmu=int(row.numPmu)
    prediction= model.predict(row)
    result=prediction[0]==1
    return numPmu,result,prediction

if __name__=='__main__':
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")
    
    save_to_file=True
    print_confusion_matrix=True
    print_training_score=True
    print_result=False
    training_files={'trot attele':'participants_trot_attele','plat':'participants_plat','trot monte':'participants_trot_monte','obstacle':'participants_obstacle'}
    # training_files=['participants_trot_attele']  

    output={'date':[],'reunion':[],'course':[],'nom':[],'rapport':[],'numPmu':[],'state':[]}    
    output_columns=['date','reunion','course','specialite','nom','rapport','numPmu','state','resultat_place','resultat_rapport','gain_brut','gain_net']
    output_df=pd.DataFrame(columns=output_columns)
    for key,file in training_files.items():
        try:
            logging.info(f"Start prediction for {key}")
            features,targets=load_file(f"{file}.csv")
            to_predict,courses,chevaux=load_file(f"to_predict_{file}.csv",is_predict=True)
            model,features_train, features_test, targets_train, targets_test =train(features,targets,shuffle=True)
            if print_confusion_matrix:
                logging.info("/"*100)
                logging.info("Confusion Matrix")
                logging.info(confusion_matrix(targets_test,model.predict(features_test)))
            if print_training_score:
                logging.info("*"*50)
                logging.info(f"{key} score: {model.score(features_test, targets_test)}")
            # nb_courses=courses.shape[1]
            # for k in range(nb_courses):
            #     r,c=courses.iloc[k].reunion,courses.iloc[k].course
                # print(courses.iloc[k].reunion)
            for course in courses.iterrows():
                x = np.asarray(course[1]).reshape(1,len(course[1]))
                r,c=x[0,0],x[0,1]
                participants_=to_predict[(to_predict['reunion']==r) & (to_predict['course']==c)]
                logging.info(f"Try to predict some Number from Reunion {r} Course {c}")
                # print(f"Calculate prediction For Reunion {r}/{c}")
                participants=participants_[NUMERICAL_FEATURES+  CATEGORICAL_FEATURES+CALCULATED_FEATURES]
                # nb_participants=participants.shape[0]
                # features_couts=participants.shape[1]
                res=participants.assign(place=model.predict(participants),
                                    reunion=r,
                                    course=c,
                                    state='place',
                                    nom=participants_['nom'],
                                    date=participants_['date'],
                                    specialite=key,
                                    resultat_place=0,
                                    resultat_rapport=0,
                                    gain_brut=0,
                                    gain_net=0)
                res=res.loc[res['place']==1][output_columns]
                if print_result:
                    nb_res=res.shape[1]
                    for z in range(nb_res):
                        t=res.iloc[z]
                        print(f"R{r}/C{c} -> {t.nom}[{t.rapport}] {t.numPmu} placé" )
                output_df=output_df.append(res.copy())
                print(output_df.shape)
                # for i in range(nb_participants):
                #     participant=participants.iloc[i]
                    
                #     p=pd.DataFrame( participant.values.reshape(1,features_couts),columns=participants.columns)
                #     # print(f"Calculate chance to be placed for {p.numPmu}")
                #     detail=participants_.iloc[i]
                #     numPmu,result,prediction=predict_place(model,p)
                #     if result:
                #         output['date'].append(detail.date)
                #         output['reunion'].append(r)
                #         output['course'].append(c)
                #         output['nom'].append(detail.nom)
                #         output['rapport'].append(participant.rapport)
                #         output['numPmu'].append(numPmu)
                #         output['state'].append('place')
                #         print(f"R{r}/C{c} -> {detail.nom}[{participant.rapport}] {numPmu} placé" )
        except FileNotFoundError:
            pass
        except Exception as ex:
            logging.warning(ex)
    if save_to_file:
        output_df.to_csv(f"predicted.csv",header=True,sep=";",mode='w')
        output_df.to_html(f"predicted.html")
        # pd.DataFrame.from_dict(output).to_csv(f"predicted.csv",header=True,sep=";",mode='a')