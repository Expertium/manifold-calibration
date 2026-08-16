@echo off
cd /d C:\Users\Andrew\manifold-calibration
python -u scrape.py >> C:\Users\Andrew\manifold-calibration\scrape.log 2>&1
echo SCRAPE_DONE_EXIT_%ERRORLEVEL% >> C:\Users\Andrew\manifold-calibration\scrape.log
python -u scrape_topics.py >> C:\Users\Andrew\manifold-calibration\topics.log 2>&1
echo TOPICS_DONE_EXIT_%ERRORLEVEL% >> C:\Users\Andrew\manifold-calibration\topics.log
echo ALL_DONE >> C:\Users\Andrew\manifold-calibration\allscrape.log
