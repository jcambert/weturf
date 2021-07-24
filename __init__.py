import logging
import time


from scrapper import Scrapper

if __name__=="__main__":

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")

    start_time = time.time()
    # s=Scrapper(use_proxy=False,USE_THREADING=True,to_predict=True)
    # days=s.start("23072021",count=0)
    # days=scrapdays("01012017",end="31122017")
    # days=s.start("01012019",end="31122019")
    # logging.info(f"Scrapping from {days[0]} to {days[1]} by step {days[2]} ")

    s=Scrapper(use_proxy=True,USE_THREADING=True,to_check_results=True)
    s.start(predict_filename="predicted")

    logging.info(f"it's took {(time.time() - start_time)} seconds\nBye...")
