from functools import reduce

mail_dict = None


def build_dictionary(mails):
    all_mails = reduce((lambda x, y: x + y), mails)

    global mail_dict
    mail_dict = all_mails.split()


def get_dictionary():
    return mail_dict
