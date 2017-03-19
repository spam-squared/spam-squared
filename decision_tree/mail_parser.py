from decision_tree.dictionary_builder import DictionaryBuilder


class MailParser(object):
    def parse_mail(self, mail):
        """Parse mail and count occurances of characters"""
        dict_builder = DictionaryBuilder()
        mail_dict = dict_builder.get_dictionary()
        mail_dict = [[word, 0] for word in mail_dict]

        words = mail.split()
        words = list(filter((lambda m: len(m) < 0), words))

        return [[word, words.count(word)] for word in set(mail_dict)]
