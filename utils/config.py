# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path as osp


class NoneDict(dict):
    def __missing__(self, key):
        return None

    def __getattr__(self, item):
        return self.get(item, None)


def dict_to_none_dict(instance):
    if isinstance(instance, dict):
        new_dict = dict()
        for k, v, in instance.items():
            new_dict[k] = dict_to_none_dict(v)
        return NoneDict(**new_dict)
    elif isinstance(instance, list):
        return [dict_to_none_dict(sub) for sub in instance]
    elif isinstance(instance, tuple):
        return (dict_to_none_dict(sub) for sub in instance)
    else:
        return instance


def parse_yaml(fp):
    assert osp.isfile(fp), f'{fp}: Not a file.'
    import yaml
    return dict_to_none_dict(yaml.safe_load(open(fp, encoding='UTF-8')))