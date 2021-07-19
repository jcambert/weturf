import json
import requests
import pandas as pd
import numpy as np
import logging
import time
import concurrent.futures,threading
from os import error, path
from datetime import datetime,  timedelta,date
proxies = {
    'http': 'socks5://localhost:9050',
    'https': 'socks5://localhost:9050'
}
PMU_MIN_DATE='01032013'
PMU_DATE_FORMAT='%d%m%Y'
USE_PROXY=True
USE_THREADING=True
TO_PREDICT=True
PREDICT_FILENAME_PREFIX="to_predict_"
prg_url="https://online.turfinfo.api.pmu.fr/rest/client/1/programme/%s?meteo=false&specialisation=INTERNET"
ptcp_url="https://online.turfinfo.api.pmu.fr/rest/client/1/programme/%s/R%s/C%s/participants?specialisation=INTERNET"
courses_filename="courses.csv"
ptcp_filename="%sparticipants_%s.csv"

def get_yesterday(fmt=False):
    yesterday = datetime.now() - timedelta(1)
    if fmt:
        return yesterday.strftime(fmt)
    return yesterday

def get_pmu_date(d):
    if isinstance(d,str):
        d=d.replace('/','')
        if len(d)==8:
            d=date(int(d[-4:]),int(d[2:4]),int(d[:2]))
        elif len(d)==6:
            century=int(datetime.now().year/100)*100
            d=date(century+int(d[-2:]),int(d[2:4]),int(d[:2]))
        else:
            raise ValueError("PMU Date string must match 010221 or 01022021")
    if isinstance(d,date):
        return d.strftime(PMU_DATE_FORMAT)
    raise ValueError(f"PMU date {d} is not supported format")

def get_date_from_pmu(d):
    if not isinstance(d,str) and len(d)!=8:
        raise ValueError(f"{d} must be a pmu date string format")
        
    return date(int(d[-4:]),int(d[2:4]),int(d[:2]))

yesterday,_yesterday=get_yesterday('%d%m%Y'),get_yesterday('%Y%m%d')

def request(url,**kwargs):
    use_proxy= kwargs['use_proxy'] if 'use_proxy' in kwargs else USE_PROXY
    if use_proxy:
        return requests.get(url, proxies=proxies)
    else:
        return requests.get(url)

def origins():
    url = 'http://httpbin.org/ip'
    o_p=json.loads(request(url,use_proxy=True).text)['origin']
    o_o=json.loads(request(url,use_proxy=False).text)['origin']
    return ( o_p,o_o)


def save(df,filename,mode='a'):
    writeHeader=not path.exists(filename) or mode=='w'
    df.to_csv(filename,sep=";",na_rep='',mode=mode,index=False,header=writeHeader)

def get_reunions(date):
    logging.info(f"Get Reunion:{prg_url%date}")
    
    try:
        resp=request(prg_url % date)
        if resp.status_code!=200:
            return False
        df=pd.DataFrame(resp.json())
        return df
    except:
        return False

def get_participants(reunion,course,date):
#     print(ptcp_url % (date,reunion,course))
    resp=request(ptcp_url % (date,reunion,course))
    df=pd.DataFrame(resp.json()['participants'])
    return df

def __scrap_participants(day,course,sub,result=False):
    subdf_ptcp=get_participants(course['numReunion'],course['numExterne'],day)
    subdf_ptcp = subdf_ptcp[subdf_ptcp['dernierRapportDirect'].notna()]
    subdf_ptcp['date']=get_date_from_pmu(day)
    subdf_ptcp['reunion']=course['numReunion']
    subdf_ptcp['course']=course['numExterne']
    subdf_ptcp['hippo_code']=sub['hippodrome']['code']
    subdf_ptcp['hippo_nom']=sub['hippodrome']['libelleCourt']
    subdf_ptcp['distance']= course['distance']
    subdf_ptcp['distanceUnit']= course['distanceUnit']
    
    # TODO CHECK
    # if 'gainsParticipant' in subdf_ptcp and not 'gainsCarriere' in subdf['gainsParticipant']:
    #     subdf_ptcp=subdf_ptcp.assign(gain_carriere=[0])
    subdf_ptcp=subdf_ptcp.assign(gain_carriere=[value['gainsCarriere'] for value in subdf_ptcp['gainsParticipant']])
    subdf_ptcp=subdf_ptcp.assign(gain_victoires=[value['gainsVictoires'] for value in subdf_ptcp['gainsParticipant']])
    subdf_ptcp=subdf_ptcp.assign(gain_places=[value['gainsPlace'] for value in subdf_ptcp['gainsParticipant']])
    subdf_ptcp=subdf_ptcp.assign(gain_annee_en_cours=[value['gainsAnneeEnCours'] for value in subdf_ptcp['gainsParticipant']])
    subdf_ptcp=subdf_ptcp.assign(gain_annee_precedente=[value['gainsAnneePrecedente'] for value in subdf_ptcp['gainsParticipant']])

    subdf_ptcp=subdf_ptcp.assign(rapport=[value['rapport'] for value in subdf_ptcp['dernierRapportDirect']])

    if not 'placeCorde' in subdf_ptcp:
        subdf_ptcp['placeCorde']=0
    if not 'handicapValeur' in subdf_ptcp:
        subdf_ptcp['handicapValeur']=0
    if not 'handicapPoids' in subdf_ptcp:
        subdf_ptcp['handicapPoids']=0
    if not 'deferre' in subdf_ptcp:
        subdf_ptcp['deferre']=0
    if not 'handicapDistance'   in subdf_ptcp:
        subdf_ptcp['handicapDistance']=0     
    col_ex=subdf_ptcp.columns.tolist()
    col_to=['date','reunion','course','hippo_code','hippo_nom', 'nom','numPmu','rapport','age','sexe','race','statut','oeilleres','deferre','indicateurInedit','musique','nombreCourses','nombreVictoires','nombrePlaces','nombrePlacesSecond','nombrePlacesTroisieme','ordreArrivee','distance','handicapDistance','gain_carriere'	,'gain_victoires'	,'gain_places'	,'gain_annee_en_cours',	'gain_annee_precedente','placeCorde','handicapValeur','handicapPoids']
    for col in filter(lambda x: x not in col_ex ,col_to) :
        logging.warning(f"{col} does not exist in dataframe")
    
    if not 'ordreArrivee' in subdf_ptcp:
        subdf_ptcp['ordreArrivee']=0
    subdf_ptcp=subdf_ptcp[col_to]
    if isinstance(result,list):
        result.append(subdf_ptcp)
        logging.info(f"END REUNION {course['numReunion']}/{course['numExterne']}")
    else:
        return subdf_ptcp

def scrap(day,save_mode='a'):
    day=get_pmu_date(day)
    print('start ',day)
    courses=[]
    courses_reunions=[]
    participants={'TROT_MONTE':[],'TROT_ATTELE':[],'PLAT':[],'OBSTACLE':[]}
    df_reunions=get_reunions(day)
    if isinstance(df_reunions,bool) and not df_reunions:
        return
    df_reunions=df_reunions['programme']['reunions']
    for r_index,reunion in enumerate(df_reunions):
        sub=pd.DataFrame.from_dict(reunion,orient="index")[0]
        subdf=pd.json_normalize(sub['courses'],max_level=1)

        threads = list()
        for c_index,course in subdf.iterrows():
            subdf_ptcp=None
            specialite=course['specialite']

            if USE_THREADING:
                logging.info(f"START REUNION {course['numReunion']}/{course['numExterne']}")
                logging.debug("Main    : create and start thread %d.", c_index)
                x = threading.Thread(target=__scrap_participants, args=(day,course,sub,participants[specialite]))
                threads.append(x)
                x.start()
            else:

                subdf_ptcp=__scrap_participants(day,course,sub)
            # subdf_ptcp=subdf_ptcp.drop(['eleveur','nomPere','nomMere','nomPereMere','gainsParticipant','dernierRapportDirect','dernierRapportReference','urlCasaque','commentaireApresCourse','distanceChevalPrecedent','robe'],axis=1,errors='ignore')

                participants[specialite].append(subdf_ptcp)
                logging.info(f"END REUNION {course['numReunion']}/{course['numExterne']}")
            
        for index, thread in enumerate(threads):
            logging.debug("Main    : before joining thread %d.", index)
            thread.join()
            logging.debug("Main    : thread %d done", index)
    # df_courses=pd.concat(courses)
    # save(df_courses,courses_filename,'w')
    for spec in participants:
        if len(participants[spec])>0:
            df_participants=pd.concat(participants[spec])
            save(df_participants,ptcp_filename % (PREDICT_FILENAME_PREFIX if TO_PREDICT else '',spec.lower()),save_mode)

    logging.info(f"End scrapping day:{day}")
    
def scrapdays(start,**kwargs):
    start=get_pmu_date(start)
    current=get_date_from_pmu( start)
    step=int(kwargs['step']) if 'step' in kwargs else 1
    end=get_date_from_pmu( kwargs['end']) if 'end' in kwargs  else ( current+timedelta( int(kwargs['count'])) if 'count' in kwargs else get_date_from_pmu(yesterday))
    sleep=int(kwargs['sleep']) if 'sleep' in kwargs else 500

    while current<=end:
        try:
            scrap(current, save_mode='a' if not TO_PREDICT else 'w' )
            time.sleep(sleep/1000)
            current=current+ timedelta(step)
        except Exception  as ex:
            logging.warn(ex,exc_info=True)
            logging.warn(f"an error happened while scrap {current}. go to next day")
            current=current+ timedelta(step)
    return (start,current,step)
def test_exist(start=None,**kwargs):
    
    start=get_pmu_date(datetime.now()) if not start else get_pmu_date(start)
    
    current=get_date_from_pmu( start)
    step=int(kwargs['step']) if 'step' in kwargs else 1
    end=get_date_from_pmu( kwargs['end'] ) if 'end' in kwargs  else False
    sleep=int(kwargs['sleep']) if 'sleep' in kwargs else 2000
    while True:
        try:
            resp=request(prg_url % get_pmu_date( current))
            exist=resp.status_code== 200
            if not exist:
                break
            current=current- timedelta(step)
            if end and current<=end:
                break
            time.sleep(sleep/1000)
        except:
            exist=False
            break
    return (exist,current)

if __name__=="__main__":

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")
    if USE_PROXY:
        logging.info('Start scrapping with proxy')
    if USE_THREADING:
        logging.info('Start scrapping with threading')

    if USE_PROXY:
        o=origins()
        if o[0]==o[1]:
            raise AssertionError(f"When using proxy {o[0]} must be different than {o[1]} \ensure you are set proxy to Internet Options")
    
    to_predict=True
    # scrap('09022020')
    start_time = time.time()
    # days=scrapdays("31012016",count=31,step=1,sleep=500)#LAST 31122016
    days=scrapdays("01012017",end="30062017")
    # days=scrapdays("19072021",count=0)
    logging.info(f"Scrapping from {days[0]} to {days[1]} by step {days[2]} ")
    logging.info(f"it's took {(time.time() - start_time)} seconds\nBye...")
    # test_exist('01042013',step=1)