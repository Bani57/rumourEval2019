from dependencies import c_pickle


def save_object(obj, filepath):
    file = open(filepath, 'wb')
    c_pickle.dump(obj, file)
    file.close()


def load_object(filepath):
    file = open(filepath, 'rb')
    obj = c_pickle.load(file)
    file.close()
    return obj
