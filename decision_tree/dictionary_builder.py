from functools import reduce


class DictionaryBuilder(object):
    mail_dict = None

    def build_dictionary(self, mails):
        all_mails = reduce((lambda x, y: x + y), mails)
        self.mail_dict = all_mails.split()

    def get_dictionary(self):
        return self.mail_dict
