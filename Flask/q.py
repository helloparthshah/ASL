import autocomplete


def getpred(s):
    return autocomplete.split_predict(s)


autocomplete.load()

print(getpred('The ap'))
