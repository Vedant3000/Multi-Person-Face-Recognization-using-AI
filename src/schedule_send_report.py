'''This module schedules the sending of email reports'''

import time
import schedule
from database_pandas import store_dataframe_in_csv, purge_dataframe
from send_mail import prepare_and_send_email
from parameters import EMAIL_SENDER, \
                       EMAIL_RECEIVER, \
                       EMAIL_SUBJECT, \
                       EMAIL_BODY, \
                       REPORT_PATH, \
                       EMAIL_SEND_TIME, \
                       EMAIL_SEND_WAIT_DURATION
from custom_logging import logger


def save_report_send_email():
    '''This function saves the report and sends email'''
    logger.info(f'Saving report and sending email at the scheduled time at: {EMAIL_SEND_TIME}')
    store_dataframe_in_csv()
    prepare_and_send_email(EMAIL_SENDER, EMAIL_RECEIVER, EMAIL_SUBJECT, EMAIL_BODY, REPORT_PATH)
    # To-do
    # Uncomment following statement only after making sure that email is correctly sent and report is saved
    # Otherwise data may be lost
    #purge_dataframe()


def schedule_send_mail():
    '''This function schedules the sending of email reports'''  
    schedule.every().day.at(EMAIL_SEND_TIME).do(save_report_send_email)
    while True:
        schedule.run_pending()
        time.sleep(EMAIL_SEND_WAIT_DURATION) # EMAIL_SEND_WAIT_DURATION seconds is a hyperparameter