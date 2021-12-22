#
#  Copyright 2019-2020 Viens Consulting, LLC. All Rights Reserved.
#
import os

import boto3
import json
import logging
import sms_spam_classifier_utilities

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Character encoding for reply email
REPLY_CHARSET = "utf-8"
REPLY_TEMPLATE = {
  "from_sender": "Deepak Dwarakanath<hw3@viegco.com>",
  "to_recipient": "Steve Viens<steve@viens.net>",
  "subject": "We received your email sent at [EMAIL_RECEIVE_DATE] with the subject [EMAIL_SUBJECT]",
  "text_body": "Here is a 240 character sample of the email body: [EMAIL_BODY]\n\nThe email was categorized as [CLASSIFICATION] with a [CLASSIFICATION_CONFIDENCE_SCORE]% confidence.",
  "html_body": "<html><head></head><body><p>Here is a 240 character sample of the email body:</p><p>[EMAIL_BODY]</p><p>The email was categorized as [CLASSIFICATION] with a [CLASSIFICATION_CONFIDENCE_SCORE]% confidence.</p></body></html>"
}


def lambda_handler(event, context):

    # Print formatted JSON event object to log
    logger.info('event={}'.format(json.dumps(event, indent=2, sort_keys=False)))

    # Get the Prediction Engine endpoint environment variable
    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    logger.info('Prediction Engine Endpoint={}'.format(ENDPOINT_NAME))

    # ################################## #
    # TODO: DO YOUR SAGEMAKER WORK HERE! #
    # ################################## #

    # These values are used in email reply string substitution below
    orig_sender = 'Steve Viens<steve@viens.net>'
    receive_date = 'ORIG-DATE'
    orig_subject = 'ORIG-SUBJECT'
    orig_body = 'ORIG-BODY'
    msg_class = 'MSG-CLASS'
    msg_score = 'MSG-SCORE'

    # Reply email message string substitution
    reply = REPLY_TEMPLATE

    subject = reply.get('subject')
    subject = subject.replace('[EMAIL_RECEIVE_DATE]', receive_date)
    subject = subject.replace('[EMAIL_SUBJECT]', orig_subject)

    htmlBody = reply.get('html_body')
    htmlBody = htmlBody.replace('[EMAIL_BODY]', orig_body)
    htmlBody = htmlBody.replace('[CLASSIFICATION]', msg_class)
    htmlBody = htmlBody.replace('[CLASSIFICATION_CONFIDENCE_SCORE]', msg_score)

    textBody = reply.get('text_body')
    textBody = textBody.replace('[EMAIL_BODY]', orig_body)
    textBody = textBody.replace('[CLASSIFICATION]', msg_class)
    textBody = textBody.replace('[CLASSIFICATION_CONFIDENCE_SCORE]', msg_score)

    # Construct reply email
    reply['to_recipients'] = orig_sender
    reply['subject'] = subject
    reply['html_body'] = htmlBody
    reply['text_body'] = textBody

    # Send report via email (SES)
    try:

        logger.info('Reply email to send via SES: {}'.format(json.dumps(reply, indent=2, sort_keys=False)))
        reply_to_sender(reply)

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


def reply_to_sender(email_msg):

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
    response = client.send_raw_email(
        RawMessage={
            'Data': msg.as_string(),
        }
    )

    return response
