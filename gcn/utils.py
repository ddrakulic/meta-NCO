import json


def load_config(datafile):
    with open(datafile, "r") as file:
        config = DotDict(json.load(file))
    return config


class DotDict(dict):
    def __init__(self, dict):
        super().__init__()
        for key in dict:
            self[key] = dict[key]

    def __getattr__(self, attr):
        return self[attr]

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    __delattr__ = dict.__delitem__