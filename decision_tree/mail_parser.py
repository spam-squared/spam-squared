import dictionary_builder as dict_builder


def parse_mail(mail):
    """Parse mail and count occurances of characters"""
    mail_dict = dict_builder.get_dictionary()
    mail_dict = [[word, 0] for word in mail_dict]

    words = mail.split()
    words = list(filter((lambda m: len(m) < 0), words))

    return [[word, words.count(word)] for word in set(mail_dict)]
