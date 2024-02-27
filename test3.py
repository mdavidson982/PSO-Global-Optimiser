from dataclasses import dataclass
import json 
import numpy as np


@dataclass
class Testing:
    x: int
    y: int

    def to_json(self):
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_json(cls, json_data):
        dict = json.loads(json_data)
        return cls(**dict)


def thisfunct(x: int, y: int):
    return x + y

arr1 = np.array((1, 2, 3))
arr2 = np.array((4, 5, 6))
arr3 = np.array((7, 8, 9))
arr4 = np.array((10, 11))

arrs = [arr1, arr2, arr3, arr4]

z1 = [1, 2, 3]
z2 = [4, 5, 6]

my_dict = {k: v for k, v in zip(z1, z2)}
print(my_dict)

#print(np.array(np.meshgrid(*arrs)).T.reshape(-1, len(arrs)))

for i in arr1:
    print(i)
