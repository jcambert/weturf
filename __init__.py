import logging
from predicter import Predicter
import time
import sys

from scrapper import HistoryScrapper, ResultatScrapper, Scrapper, ToPredictScrapper

if __name__=="__main__":

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

    start_time = time.time()
    # s=Scrapper(use_proxy=False,USE_THREADING=True,to_predict=True)
    # days=s.start("23072021",count=0)
    # days=scrapdays("01012017",end="31122017")
    # days=s.start("01012019",end="31122019")
    # logging.info(f"Scrapping from {days[0]} to {days[1]} by step {days[2]} ")

    # s=Scrapper(use_proxy=True,USE_THREADING=True,to_check_results=True)
    # s.start(predict_filename="predicted")

    # scrapper=ResultatScrapper(use_proxy=True,use_threading=True,test=True)
    # scrapper=HistoryScrapper(use_proxy=True,use_threading=True,test=True)
    if sys.argv[1] in  ["benchmark"]:
        predicter=Predicter(use_threading=False,train_size=4000,test_size=0.2)
        predicter.benchmark()

    if sys.argv[1] in ["scrapper","all"]:
        scrapper=ToPredictScrapper()
        scrapper.start()

    if sys.argv[1] in ["predicter","all"]:
        predicter=Predicter(print_result=True,test_size=0.3)
        predicter.start(training_files= {'trot attele':'trot_attele'})

    if sys.argv[1] == "history":
        start=sys.argv[2]
        end=sys.argv[3]
        scrapper=HistoryScrapper()
        scrapper.start(start=start,end=end)

    if sys.argv[1]=="l_curve":
        predicter=Predicter(use_threading=False,test=True,train_size=4000,test_size=0.2)
        values=predicter.start(training_files= {'trot attele':'trot_attele'}, learning_curve=True)
        print(values)
        print(values['trot attele'])
        N=values['trot attele'][0][0]
        train_score=values['trot attele'][0][1]
        val_score=values['trot attele'][0][2]
        pass
    logging.info(f"it's took {(time.time() - start_time)} seconds\nBye...")
