from dataclasses import dataclass as _dataclass, fields

"""
Wrapper for dataclass that allows checking if a field exists
with syntax if 'field_name' in object.

Example:

@dataclass
class Person:
    name: str
    age: int
    city: str

person = Person(name="Alice", age=30, city="New York")
print("name" in person)   # True
print("foo" in person)    # False
"""

def dataclass(cls=None, **kwargs):
    def add_contains(cls):
        result = _dataclass(cls, **kwargs)
        _field_names = tuple(f.name for f in fields(result))
        def __contains__(self, key):
            return key in _field_names
        result.__contains__ = __contains__
        return result
    if cls is None:
        return add_contains
    return add_contains(cls)
