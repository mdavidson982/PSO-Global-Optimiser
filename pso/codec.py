"""
File which handles json serialization/deserialization for various dataclasses.
"""

from io import TextIOWrapper
import json
import numpy as np
from . import psodataclass as dc

_DATATYPE = "datatype"
_DATA = "data"

class _DataClassEncoder(json.JSONEncoder):
    """Class which handles encoding of dataclass objects"""
    def default(self, o):
        # Convert numpy arrays to lists, which has proven to be the easiest way to jsonize
        if type(o) == np.ndarray:
            return o.tolist()
        elif type(o)in list(dc.DATACLASSES.values()):
            dataclass_dict = {
                _DATATYPE: type(o).__name__,
                _DATA: o.__dict__
            }
            return dataclass_dict
        else:
            return o.__dict__

def _decoder_func(replacement_rules: dict[str, any]) -> any:
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
    return json.dumps(obj = obj, cls = _DataClassEncoder)

def dataclass_to_json_file(obj, file: TextIOWrapper):
    json.dump(obj = obj, fp = file, cls = _DataClassEncoder)

def json_to_dataclass(jsonstring: str):
    dataclass_name = json.loads(jsonstring)[_DATATYPE]
    dataclass = dc.DATACLASSES[dataclass_name]

    """Convert a jsonized dataclass string to an object"""
    hook_function = _decoder_func(dataclass.decode_json_hooks)
    return dataclass(**json.loads(jsonstring, object_hook=hook_function)[_DATA])

def json_file_to_dataclass(file: TextIOWrapper):
    # Fetch the dataclass type from the top level of the json file, and return its class type
    dataclass_name = json.load(file)[_DATATYPE]
    dataclass = dc.DATACLASSES[dataclass_name]
    file.seek(0)
    # Return its respective hooks, needed for decoding
    hook_function = _decoder_func(dataclass.decode_json_hooks)
    dataclass_dict = json.load(file, object_hook=hook_function)
    print(dataclass_dict[_DATA])
    return dataclass(**dataclass_dict[_DATA])

if False:
    mypath = os.path.join("benchmark_tests", "configs")
    files = os.listdir(mypath)
    for funct in tf.TESTFUNCTSTRINGS:
        functpath = os.path.join(mypath, funct)
        

        with open(os.path.join(functpath,"domain_data.json"), "r") as file:
            
            z = codec.json_file_to_dataclass(file)
        z.optimum = np.zeros(30)
        z.bias = 0
        with open(os.path.join(functpath, "domain_data.json"), "w+") as file:
            codec.dataclass_to_json_file(z, file)