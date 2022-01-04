import os, io, sys

import boto3
import json, csv
import logging
import sms_spam_classifier_utilities
import re

import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from botocore.exceptions import ClientError

import numpy as np
from sms_spam_classifier_utilities import one_hot_encode, vectorize_sequences
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Vocabulary Size for one_hot_encode, vectorize_sequences
vocabulary_length = 9013
# Character encoding for reply email
REPLY_CHARSET = "utf-8"
REPLY_TEMPLATE = {
  "from_sender": "Deepak EmailServer<dd2676@deepak-dwarakanath.com>",
  "to_recipient": "Deepak Dwarakanath<dd2676@columbia.edu>",
  "subject": "We received your email sent at \"[EMAIL_RECEIVE_DATE]\" with the subject \"[EMAIL_SUBJECT]\"",
  "text_body": "Here is a 240 character sample of the email body: \"[EMAIL_BODY]\"\n\nThe email was categorized as \"[CLASSIFICATION]\" with a [CLASSIFICATION_CONFIDENCE_SCORE]% confidence.",
  "html_body": "<html><head></head><body><p>Here is a 240 character sample of the email body:</p><p>[EMAIL_BODY]</p><p>The email was categorized as [CLASSIFICATION] with a [CLASSIFICATION_CONFIDENCE_SCORE]% confidence.</p></body></html>"
}


def lambda_handler(event, context):
    
    # NEED TO INITIALIZE THE FOLLOWING EVERY TIME THE EVENT HANDLER IS CALLED.  FOR SOME REASON, REPLY_TEMPLATE IS BEING OVERWRITTEN.  MAYBE THIS IS A BUG IN THE LAMBDA SERVICE.
    # Vocabulary Size for one_hot_encode, vectorize_sequences
    # vocabulary_length = 9013
    # # Character encoding for reply email
    # REPLY_CHARSET = "utf-8"
    # REPLY_TEMPLATE = {
    #   "from_sender": "Deepak EmailServer<dd2676@deepak-dwarakanath.com>",
    #   "to_recipient": "Deepak Dwarakanath<dd2676@columbia.edu>",
    #   "subject": "We received your email sent at \"[EMAIL_RECEIVE_DATE]\" with the subject \"[EMAIL_SUBJECT]\"",
    #   "text_body": "Here is a 240 character sample of the email body: \"[EMAIL_BODY]\"\n\nThe email was categorized as \"[CLASSIFICATION]\" with a [CLASSIFICATION_CONFIDENCE_SCORE]% confidence.",
    #   "html_body": "<html><head></head><body><p>Here is a 240 character sample of the email body:</p><p>[EMAIL_BODY]</p><p>The email was categorized as [CLASSIFICATION] with a [CLASSIFICATION_CONFIDENCE_SCORE]% confidence.</p></body></html>"
    # }
    # Get the Prediction Engine endpoint environment variable
    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    # logger.info('Prediction Engine Endpoint={}'.format(ENDPOINT_NAME))
    # Print formatted JSON event object to log
    t1 = time.time()
    print("##### START_TIME = {} #####".format(t1-t1))
    print("EVENT:\n", event)
    print("\nREPLY_TEMPLATE 1:\n", REPLY_TEMPLATE)
    mybucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    # logger.info('event={}'.format(json.dumps(event, indent=2, sort_keys=False)))
    mybucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    # mybucket = 'deepak-email-bucket'
    # key = "llgd185fdqlg4qd5sls6rmgo5j6oph13ho981fg1"
    # print(mybucket, key)
    
    s3 = boto3.client('s3')
    raw_email = s3.get_object(
        Bucket=mybucket,
        Key=key)
    raw_email_body = raw_email.get('Body').read().decode()
    origin_message = email.message_from_string(raw_email_body)
    email_regex =  r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # if re.search(email_regex, origin_message['From']) is not None:
    #     email_indices = re.search(email_regex, origin_message['Return-Path']).span()
    #     origin_sender = origin_message['Return-Path'][email_indices[0]:email_indices[-1]]
    # else:
    #     raise Exception("{} is not a valid email address!!!".format(message['Return-Path']))
    origin_sender = origin_message['From']
    
    # print(origin_sender)
    receive_date = origin_message['Date']
    origin_subject = origin_message['Subject']
    body_list = [p.get_payload() for p in origin_message.get_payload()]
    origin_body = body_list[0]
    # logger.info(body_list)
    print("\n----------\n")
    print("Receive Date: {}".format(receive_date))
    print("Origin Subject: {}".format(origin_subject))
    print("Origin Body: {}".format(origin_body))
    print("\n----------\n")
    # logger.info(origin_body)
    
    sagemaker_client = boto3.client('sagemaker')
    ep_list = sagemaker_client.list_endpoints()['Endpoints']
    
    for ep in ep_list:
        if 'sms-spam-classifier' in ep['EndpointName'] and ep['EndpointStatus']=='InService':
            endpoint_name = ep['EndpointName']
        else:
            raise Exception("No valid Sagemaker endpoint available.")
    print("EndpointName = ", endpoint_name)
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    def np2csv(arr):
        csv = io.BytesIO()
        np.savetxt(csv, arr, delimiter=',', fmt='%g')
        return csv.getvalue().decode().rstrip()

    origin_body_no_escape = origin_body.replace('\n',' ').replace('\r',' ')
    
    # print(origin_body_no_escape)
    # logger.info(origin_body_no_escape)
    origin_body_no_escape_list = [origin_body_no_escape]
    one_hot_origin_body = one_hot_encode(origin_body_no_escape_list, vocabulary_length)
    vectorized_origin_body = vectorize_sequences(one_hot_origin_body, vocabulary_length)
    # print("One hot ", one_hot_origin_body)
    # print("vectorized_origin_body shape = ", vectorized_origin_body.shape)
    # print("----------")
    origin_body_bytestream = np2csv(vectorized_origin_body)
    a = len(origin_body_bytestream)
    b = type(origin_body_bytestream)
    # print("origin_body_bytestream length = {}, type = {}".format(a,b)) # .format(len(origin_body_bytestream),type(origin_body_bytestream)))
    
    response = sagemaker_runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, 
                                  ContentType='text/csv',
                                  Body=origin_body_bytestream)
    result = json.loads(response['Body'].read().decode())
    prediction_score = result['predictions'][0]['score']
    spamOrHam = result['predictions'][0]['predicted_label']
    # print("Score = {}, SpamOrHam={}".format(prediction_score, spamOrHam))
    
    msg_score = str(prediction_score)
    if spamOrHam==1:
        msg_class = 'SPAM'
    else:
        msg_class = 'NOT SPAM'
    
    # Reply email message string substitution
    reply = REPLY_TEMPLATE.copy()

    subject = reply.get('subject')
    subject = subject.replace('[EMAIL_RECEIVE_DATE]', receive_date)
    subject = subject.replace('[EMAIL_SUBJECT]', origin_subject)

    htmlBody = reply.get('html_body')
    htmlBody = htmlBody.replace('[EMAIL_BODY]', origin_body)
    htmlBody = htmlBody.replace('[CLASSIFICATION]', msg_class)
    htmlBody = htmlBody.replace('[CLASSIFICATION_CONFIDENCE_SCORE]', msg_score)

    textBody = reply.get('text_body')
    textBody = textBody.replace('[EMAIL_BODY]', origin_body)
    textBody = textBody.replace('[CLASSIFICATION]', msg_class)
    textBody = textBody.replace('[CLASSIFICATION_CONFIDENCE_SCORE]', msg_score)

    # Construct reply email
    reply['to_recipients'] = origin_sender
    reply['subject'] = subject
    reply['html_body'] = htmlBody
    reply['text_body'] = textBody
    # print("---------------------")
    # print("---------------------")
    # print(reply)
    # print("---------------------")
    # print("---------------------")
    print("##########")
    print("REPLY TEMPLATE 2:\n{}".format(REPLY_TEMPLATE))
    print("##########")
    # Send report via email (SES)
    try:

        # logger.info('Reply email to send via SES: {}'.format(json.dumps(reply, indent=2, sort_keys=False)))
        print("##########")
        print("REPLY TEMPLATE 3\n:{}".format(REPLY_TEMPLATE))
        print("##########")
        # print('Reply email to send via SES: {}'.format(json.dumps(reply, indent=2, sort_keys=False)))
        reply_to_sender(reply, t1, REPLY_TEMPLATE, REPLY_CHARSET)

    except ClientError as e:

        logger.error((e.response['Error']['Message']))
        return {
            'statusCode': 400,
            'body': json.dumps(e.response['Error']['Message'])
        }

    else:

        logger.info('Reply email sent!')
        return {
            'statusCode': 200,
            'body': json.dumps('Reply email sent!')
        }

def reply_to_sender(email_msg, t1, reply_template_in, REPLY_CHARSET):

    # Extract email properties
    from_sender = email_msg.get('from_sender', '')
    to_recipients = email_msg.get('to_recipients', '')
    cc_recipients = email_msg.get('cc_recipients', '')
    bcc_recipients = email_msg.get('bcc_recipients', '')
    subject = email_msg.get('subject', '')
    text_body = email_msg.get('text_body', '')
    html_body = email_msg.get('html_body', '')

    # Create a multipart/mixed parent container.
    msg = MIMEMultipart('mixed')

    # Add subject, from and to/cc/bcc email addresses
    msg['Subject'] = subject
    msg['From'] = from_sender
    msg['To'] = to_recipients
    msg['Cc'] = cc_recipients
    msg['Bcc'] = bcc_recipients

    # Create a multipart/alternative child container.
    msg_body = MIMEMultipart('alternative')

    # Encode the text and html content and set the character encoding.
    # Encoding to a specific character set is necessary if you're sending
    # a message with characters outside the ASCII range.
    textpart = MIMEText(text_body.encode(REPLY_CHARSET), 'plain', REPLY_CHARSET)
    htmlpart = MIMEText(html_body.encode(REPLY_CHARSET), 'html', REPLY_CHARSET)

    # Add the text and text email message body to the child container.
    msg_body.attach(textpart)
    msg_body.attach(htmlpart)

    # Attach the multipart/alternative child container to the multipart/mixed
    # parent container.
    msg.attach(msg_body)

    # Create an SES client & send email message
    client = boto3.client('ses')
    t2 = time.time()-t1
    # print("##### before sen TIME = {} #####".format(t2))
    # print('-------------------')
    # print('--------FINAL MESSAGE-----------')
    final_message = msg.as_string()
    # print(final_message)
    # print('-------------------')
    # print('-------------------')
    print("##########")
    print("REPLY TEMPLATE 4:\n{}".format(reply_template_in))
    print("##########")
    response = client.send_raw_email(
        RawMessage={
            'Data': final_message,
        }
    )
    print("FINAL TIME = {}".format(time.time()-t1))
    logger.info("RESPONSE:\n{}".format(response))
    return response

    