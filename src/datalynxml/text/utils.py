import numpy as np
import tiktoken
from urllib.parse import urlparse


def compute_similarity_score(vector1, vector2):
    """
    Computes the similarity score between two vectors using the dot product and the length of each vector.

    Args:
        vector1 (list): The first vector to be compared.
        vector2 (list): The second vector to be compared.

    Returns:
        float: The similarity score between the two input vectors.
    """

    return np.dot(vector1, vector2) / (len(vector1) * len(vector2))


def compute_cosine_similarity(vec1, vec2):
    # Compute the dot product of the two vectors
    dot_product = np.dot(vec1, vec2)

    # Compute the L2 norm (Euclidean norm) of each vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Compute the cosine similarity
    similarity = dot_product / (norm_vec1 * norm_vec2)

    return similarity


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Returns the number of tokens in a text string using a specified encoding.

    Args:
        string (str): The input text string.
        encoding_name (str): The name of the encoding to use for tokenization.

    Returns:
        int: The number of tokens in the input text string.
    """
    # encoding = tiktoken.get_encoding(encoding_name)
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def count_tokens(text):
    """
    Counts the number of tokens in a text string using the GPT-2 encoding.

    Args:
        text (str): The input text string.
    Returns:
        int: The number of tokens in the input text string.
    """

    num_tokens = num_tokens_from_string(str(text), "gpt-4")
    return num_tokens


def truncate_string(text, max_tokens, step=10):
    """
    Truncates a string to contain no more than a maximum number of tokens, by iteratively removing a given number of tokens.

    Args:
        text (str): The input string to be truncated.
        max_tokens (int): The maximum number of tokens allowed in the output string.
        step (int): The number of tokens to remove at each iteration. Defaults to 1.

    Returns:
        str: The truncated string that contains no more than `max_tokens` tokens.
    """
    num_tokens = count_tokens(text)
    while num_tokens > max_tokens:
        tokens = text.split()
        num_tokens = count_tokens(" ".join(tokens[:-step]))
        text = " ".join(tokens[:-step])
    return text


def split_string(input_string, num_splits):
    """
    Splits an input string into a specified number of parts.

    This function divides a string into `num_splits` parts, each part having an approximately equal length.
    If the string cannot be divided exactly, the last substring will be shorter and appended to the
    second-to-last substring.

    Parameters:
    input_string (str): The string to be split.
    num_splits (int): The number of parts to split the string into.

    Returns:
    list of str: A list containing the split substrings.
    """
    # Calculate the length of each split
    split_length = len(input_string) // num_splits

    # Split the input string into a list of substrings
    substrings = [
        input_string[i : i + split_length]
        for i in range(0, len(input_string), split_length)
    ]

    # If the last substring is shorter than the split length, append it to the previous substring
    if len(substrings[-1]) < split_length:
        substrings[-2] += substrings[-1]
        substrings.pop()

    return substrings
