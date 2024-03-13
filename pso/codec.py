"""
File which handles json serialization/deserialization for various dataclasses.
"""

from io import TextIOWrapper
import json
import numpy as np
from . import psodataclass as dc

class DataClassEncoder(json.JSONEncoder):
    """Class which handles encoding of dataclass objects"""
    def default(self, o):
        # Convert numpy arrays to lists, which has proven to be the easiest way to jsonize
        if type(o) == np.ndarray:
            return o.tolist()
        else:
            return o.__dict__

def decoder_func(replacement_rules: dict[str, any]) -> any:
    """Function which helps handle custom json deserialization hooks.
    See JSONDecoder object hooks.
    """
    def my_decoder_func(dict: dict[str, any]) -> any:
        for key, value in dict.items():
            if key in replacement_rules:
                dict[key] = replacement_rules[key](value)
        return dict
    return my_decoder_func

def dataclass_to_json(obj) -> str:
    """Convert a dataclass to a jsonized string"""
    return json.dumps(obj = obj, cls=DataClassEncoder)

def dataclass_to_json_file(obj, file: str | TextIOWrapper):
    """Write a jsonized dataclass to a file"""
    if type(file) == str:
        with open(file, "w+") as file:
            json.dump(obj = obj, fp = file, cls = DataClassEncoder)
    else:
        json.dump(obj = obj, fp = file, cls = DataClassEncoder)

def json_to_dataclass(jsonstring: str, dataclass: dc.Dataclass):
    """Convert a jsonized dataclass string to an object"""
    hook_function = decoder_func(dataclass.decode_json_hooks())
    return dataclass(**json.loads(jsonstring, object_hook=hook_function))

def json_file_to_dataclass(file: str | TextIOWrapper, dataclass: dc.Dataclass):
    """Load a json file into a dataclass"""
    hook_function = decoder_func(dataclass.decode_json_hooks())

    if type(file) == str:
        with open(file, "w+") as file:
            return dataclass(**json.load(file, object_hook=hook_function))
    else:
        return dataclass(**json.load(file, object_hook=hook_function))