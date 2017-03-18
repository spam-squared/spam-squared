import sendgrid
import os
from sendgrid.helpers.mail import *
from flask import Flask, request
import simplejson
app = Flask(__name__)

sg = sendgrid.SendGridAPIClient(apikey=os.environ.get('SENDGRID_API_KEY'))
from_email = apikey=os.environ.get('EMAIL')
recieverHooks = []

@app.route('/parse', methods=['POST'])
def sendgrid_parser():
  # Consume the entire email
  envelope = simplejson.loads(request.form.get('envelope'))

  # Get some header information
  to_address = envelope['to'][0]
  from_address = envelope['from']

  # Now, onto the body
  text = request.form.get('text')
  subject = request.form.get('subject')

  # Process the attachements, if any
  for reciever in recieverHooks:
      reciever(subject, from_address, text)

  return "OK"

if __name__ == '__main__':
    app.run(debug=True)

def add_reciever(reciever):
    recieverHooks.append(reciever)

def send(reciever, content, subject):
    to_email = Email(reciever)
    content_email = Content("text/plain", content)
    mail = Mail(from_email, subject, to_email, content_email)
    response = sg.client.mail.send.post(request_body=mail.get())
    return response