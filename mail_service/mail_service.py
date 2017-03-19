from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
import imaplib
import smtplib
import email
from threading import Thread
from time import sleep


class MailService(object):
    SMTP_SERVER = "smtp.gmail.com"
    FROM_EMAIL = "keith.responsum@gmail.com"
    FROM_PWD = "ILoveSpam"

    FOOTER = "\n\n\n-\n\nKeith Responsum"

    receivers = []

    def __init__(self):
        def check_mail_loop(self):
            print("Checking mail...")
            data = self.read_mail()

            if data != -1:
                for func in self.receivers:
                    func(data)

            sleep(10)
            check_mail_loop(self)

        mail_thread = Thread(target=check_mail_loop, args=(self,))
        mail_thread.start()

    def add_receiver(self, receiver):
        self.receivers.append(receiver)

    def read_mail(self):
        try:
            mail = imaplib.IMAP4_SSL(self.SMTP_SERVER)
            mail.login(self.FROM_EMAIL, self.FROM_PWD)
            mail.select('inbox')

            type, data = mail.search(None, 'ALL')
            mail_ids = data[0]

            id_list = mail_ids.split()
            first_email_id = int(id_list[0])
            latest_email_id = int(id_list[-1])

            for i in range(latest_email_id, first_email_id, -1):
                typ, data = mail.fetch(str(i), "(RFC822)")

                for response_part in data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        email_subject = msg['subject']
                        email_from = msg['from']

                        body = ""
                        if msg.is_multipart():
                            for payload in msg.get_payload():
                                body += payload.get_payload()
                        else:
                            body = msg.get_payload()

                        print('NEW EMAIL:' + '\n')
                        print('From : ' + email_from + '\n')
                        print('Subject : ' + email_subject + '\n')
                        print('Body : ' + body + '\n')

                        mail.expunge()
                        mail.close()
                        mail.logout()

                        return {
                            "mail_from": email_from,
                            "subject": email_subject,
                            "body": body
                        }

                mail.store(str(i), '+FLAGS', '\\Deleted')
                print('Removed read email')

            mail.expunge()
            mail.close()
            mail.logout()

            return -1

        except Exception as e:
            print(str(e))

    def send_mail(self, to_address, subject, body):
        fromm = self.FROM_EMAIL
        to = [to_address]

        email_text = """\
                    From: %s
                    To: %s
                    Subject: %s

%s
                    """ % (fromm, ", ".join(to), subject, body + self.FOOTER)

        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.ehlo()
            server.login(self.FROM_EMAIL, self.FROM_PWD)
            server.sendmail(fromm, to, email_text)
            server.close()

            print ('Email sent!')

        except Exception as e:
            print(str(e))
