# coding: utf-8
import numpy as np

class DataMixin(object):
    AMIDOGEN = [
        'A',
        'R',
        'N',
        'D',
        'C',
        'Q',
        'E',
        'G',
        'H',
        'I',
        'L',
        'K',
        'M',
        'F',
        'P',
        'S',
        'T',
        'W',
        'Y',
        'U',
        'V',
        'X',
        'Y',
        'Z',
    ]
    SECSTR = ['H', 'E', 'C']
    START_TOKEN = 1

    def amido2id(self, slist, mask=True):
        return [self.AMIDOGEN.index(s) + 1 if mask else self.AMIDOGEN.index(s) for s in slist]

    def sec2id(self, slist, mask=True, start_token=False):
        shift = 1 if start_token else 0
        shift = shift + 1 if mask else shift
        org_list = [self.SECSTR.index(s) + shift for s in slist]
        return [self.START_TOKEN] + org_list if start_token else org_list

    def id2amido(self, id_list, mask=True):
        shift = -1 if mask else 0
        return [self.AMIDOGEN[idx + shift] for idx in id_list]

    def id2sec(self, id_list, mask=True, start_token=False):
        shift = -1 if mask else 0
        shift = shift - 1 if start_token else shift
        return [self.SECSTR[idx + shift] for idx in id_list]

    def one_hot(self, targets, num_classes):
        res = np.eye(num_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[num_classes])
