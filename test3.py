from typing import TypeVar

T = TypeVar('T')

def add_type_hints(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        
        # Add type hints for the new method
        func.__annotations__["self"] = cls
        func.__annotations__["return"] = T
        
        # Add a docstring for the new method
        if not func.__doc__:
            func.__doc__ = "This is a dynamically added method."
        
        return func
    return decorator

@add_type_hints
class MyClass:
    def new_method(self: 'MyClass', arg: str) -> str:
        """A new dynamically added method."""
        return f"New method called with argument: {arg}"


