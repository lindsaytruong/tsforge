from typing import Any

def call_method(obj: Any, method: str, **kwargs):
    """
    Call a method on either a pandas DataFrame or a DataFrameGroupBy object.
    Raises AttributeError if method is missing.
    """
    fn = getattr(obj, method)
    return fn(**kwargs)
