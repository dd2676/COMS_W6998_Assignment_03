#
#  Copyright 2019-2020 Viens Consulting, LLC. All Rights Reserved.
#

from datetime import datetime
import predict_spam_function
import logging
import pytz
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

event = {
}

timezone = pytz.timezone('US/Eastern')
timestamp = datetime.now(timezone)
started = datetime.now(timezone)
response = predict_spam_function.lambda_handler(event, context={})
status = response.get('statusCode')
ended = datetime.now(timezone)
print("--------------------------")
print("HTTP STATUS  = {}".format(status))
print("STARTED      = {}".format(started.strftime('%Y-%m-%d %H:%M:%S %Z')))
print("ENDED        = {}".format(ended.strftime('%Y-%m-%d %H:%M:%S %Z')))
print("DURATION     = {}".format(ended - started))
