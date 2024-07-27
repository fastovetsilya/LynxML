import copy
import json
import base64
import random
from itertools import groupby
from typing import Any, List, Optional, Tuple, Union
from uuid import UUID
from decimal import Decimal
from datetime import date
from pydantic import BaseModel


class CustomEncoder(json.JSONEncoder):
    """
    Custom JSON encoder subclass that handles specific object types not supported by the default JSON encoder.

    This encoder is designed to convert complex Python data types such as Decimals, dates, bytearrays, and UUIDs into JSON serializable formats. It also properly handles `None` values.

    Inherits from `json.JSONEncoder`.
    """

    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, bytearray):
            # Convert bytearray to a base64 encoded string
            return base64.b64encode(obj).decode("utf-8")
        if isinstance(obj, UUID):  # Check if the object is a UUID
            return str(obj)  # Return the UUID as a string
        if obj is None:
            return None
        return json.JSONEncoder.default(self, obj)
    
    
def estimate_categorical_columns(data: List[Tuple]) -> List[bool]:
    """Estimates which columns are categorical based on the uniqueness ratio."""
    columns = list(zip(*data))
    categorical = []
    for col in columns:
        unique_values = len(set(col))
        total_values = len(col)
        # Assuming a column is categorical if less than 30% of its values are unique
        categorical.append(unique_values / total_values < 0.3)
    return categorical


def intelligent_truncate_data(data: List[List], truncate_max_res: int = 3) -> str:
    # Reduce data size if necessary
    if len(data) > 100000:
        data = random.sample(data, 100000)

    # Assuming estimate_categorical_columns() exists and returns a list of booleans
    is_categorical = estimate_categorical_columns(data)

    # Identify and sort columns by number of unique values (for categorical ones)
    categorical_info = [
        (i, len(set(col))) for i, col in enumerate(zip(*data)) if is_categorical[i]
    ]
    categorical_info.sort(key=lambda x: x[1])

    # Adjust the sorting of data to prioritize columns with fewer unique values
    data_sorted = sorted(
        data,
        key=lambda row: tuple(
            (val is None, val)  
            for i, val in enumerate(row)
            if i in [i for i, _ in categorical_info]
        ),
    )

    # Group by categorical columns with respect to their priority
    groups = groupby(
        data_sorted, key=lambda row: tuple(row[i] for i, _ in categorical_info)
    )

    summarized_data = [list(next(group)) for _, group in groups]
    unique_data_set = set(tuple(row) for row in summarized_data)
    additional_data_needed = len(data) - len(unique_data_set)

    for row in data:
        if additional_data_needed <= 0:
            break
        if tuple(row) not in unique_data_set:
            summarized_data.append(list(row))
            unique_data_set.add(tuple(row))
            additional_data_needed -= 1

    # Convert elements to str and encapsulate Optional use here if expected Nonetype
    summarized_str_data = [[str(item) if item is not None else None for item in row] for row in summarized_data]

    # Apply truncation based on the specified resolution limit if there are more rows than allowed
    if len(summarized_str_data) > truncate_max_res:
        return json.dumps(summarized_str_data[:truncate_max_res])  # Simple cut-off at max resolution
    else:
        return json.dumps(summarized_str_data)

class QueryData(BaseModel):
    column_names: Optional[List[str]] = None
    results: Optional[List[List[Optional[Any]]]] = None
    error: Optional[str]

class QueryDataTruncated(BaseModel):
    column_names: Optional[List[str]] = None
    results: Optional[str] = None
    error: Optional[str]

class ModelResponse(BaseModel):
    text: str
    query: Optional[str] = None
    query_data: Optional[QueryData] = None
    query_data_truncated: Optional[QueryDataTruncated] = None