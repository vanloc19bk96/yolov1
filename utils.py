import configparser
from collections import OrderedDict


def read_cfg(fname):
    config = configparser.ConfigParser(defaults=None, dict_type=MultiDict, strict=False)
    config.read(fname)
    return config


class MultiDict(OrderedDict):
    _unique = 0

    def __setitem__(self, key, val):
        if isinstance(val, dict):
            self._unique += 1
            key += '-' + str(self._unique)
        OrderedDict.__setitem__(self, key, val)
