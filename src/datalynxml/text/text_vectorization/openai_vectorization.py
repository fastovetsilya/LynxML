import os
import sys
from ast import literal_eval
from dotenv import load_dotenv
from typing import Dict, Tuple, List, Any, Optional
from pydantic import (
    BaseModel as BaseModel_v2,
    Field as Field_v2,
    PrivateAttr as PrivateAttr_v2,
)
import openai

dir_path = os.path.abspath(os.path.join(__file__, "../"))
root_path = os.path.abspath(os.path.join(__file__, "../../../.."))
sys.path.append(root_path)

from datalynxml.text.utils import count_tokens, truncate_string


class OpenAITextVectorization(BaseModel_v2):
    """
    A class to handle text vectorization using OpenAI.

    Attributes:
        openai_api_key (str): OpenAI API key
        Not set:
        text_input (str): The input text to be vectorized.
        array_output (list): The output vector after vectorizing the input text.
    """

    openai_api_key: str
    # Public attributes that are not set
    text_input: Optional[str] = Field_v2(default_factory=str)
    array_output: Optional[list] = Field_v2(default_factory=list)

    class Config:
        arbitrary_types_allowed = False

    def __init__(self, **data):
        super().__init__(**data)
        openai.api_key = self.openai_api_key

    def vectorize_text(self, text_input: str) -> list:
        """
        Vectorizes the input text using OpenAI model and returns the generated vector.

        Args:
            text_input (str): The input text to be vectorized.

        Returns:
            list: The output vector after vectorizing the input text.
        """
        self.text_input = text_input

        # Hard-code values because .ini files don't work well
        model = "text-embedding-3-small"
        model_max_input = 4000

        content = text_input.encode(encoding="ASCII", errors="ignore").decode()
        content_tokens_len = count_tokens(content)
        if content_tokens_len > model_max_input:
            print(f"Warning! Trimming content to {model_max_input} level...")
            content = truncate_string(content, model_max_input, step=500)
        response = openai.embeddings.create(input=[content], model=model)
        vector = response.data[0].embedding  # this is a normal list
        self.array_output = vector
        return vector


if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    example_text = "Customer growth"
    text_vectorization = OpenAITextVectorization(openai_api_key=openai_api_key)
    print("Text vectorization loaded successfully")
    print("Performing vectorization...")
    text_vector = text_vectorization.vectorize_text(text_input=example_text)
    print(f"Vectorization executed successfully. The vector is: {text_vector}")
    print(f"The length of the output vector is: {len(text_vector)}")
