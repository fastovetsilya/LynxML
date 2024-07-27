import anthropic
import os
import sys
import threading
import logging
from dotenv import load_dotenv
from configparser import ConfigParser
from multiprocessing import Process, Queue
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)
from typing import Dict, Tuple, List, Any, Optional
from pydantic.v1 import BaseModel, Field
from pydantic import (
    BaseModel as BaseModel_v2,
    Field as Field_v2,
    PrivateAttr as PrivateAttr_v2,
)

# root_path = os.path.abspath(os.path.join(__file__, "../../../.."))
# sys.path.append(root_path)

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)

RED_START = "\033[31m"
RED_END = "\033[0m"
COST_PER_TOKEN_PROMPT = 0.015 / 1000  # 0.01, 0.015
COST_PER_TOKEN_SAMPLED = 0.075 / 1000  # 0.03, 0.075


# Defining a completion function with retry decorator
@retry(wait=wait_exponential(multiplier=1, min=10, max=30), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=kwargs.get("api_key"),
    )
    message = client.messages.create(
        model=kwargs.get("model"),
        max_tokens=kwargs.get("max_tokens"),
        messages=kwargs.get("messages"),
    )
    return message


class AnthropicTextCompletion(BaseModel_v2):
    """
    A class for handling text completion tasks using OpenAI's GPT models.

    This class is designed to interact with OpenAI's API for generating text completions.
    It includes functionality to load configuration settings, handle API keys, and
    manage the process of sending requests and receiving responses from the API.

    Attributes:
        config_path (str): Path to the configuration file.
        openai_api_key (str): API key for accessing OpenAI's services.

    Methods:
        claude_operation: Handles the Claude model operation in a separate process.
        chat_complete_claude: Initiates the Claude model operation and manages timeouts.

    Raises:
        FileNotFoundError: If the configuration file does not exist or is empty.
    """

    config: dict
    anthropic_api_key: str

    class Config:
        arbitrary_types_allowed = False

    def __init__(self, **data):
        super().__init__(**data)

    def claude_operation(self, messages, queue, functions=None):
        try:
            # Build the request data dynamically
            request_data = {
                "api_key": self.anthropic_api_key,
                "model": self.config["model"],
                "system": [m for m in messages if m["role"] == "system"],
                "messages": [m for m in messages if m["role"] != "system"],
                "max_tokens": int(self.config["max_tokens"]),
                "temperature": float(self.config["temperature"]),
            }

            # If functions parameter is provided and is a list, add it to the request data
            if functions and isinstance(functions, list):
                request_data["functions"] = functions

            logger.debug("Generating response...")
            response = completion_with_backoff(**request_data)
            logger.debug(f"Response is: {response}")

            # Print the usage from the response
            token_usage = response.usage
            prompt_tokens = token_usage.input_tokens
            completion_tokens = token_usage.output_tokens
            total_cost = (prompt_tokens * COST_PER_TOKEN_PROMPT) + (
                completion_tokens * COST_PER_TOKEN_SAMPLED
            )
            total_cost = round(total_cost, 5)
            logger.info(
                f"""
                        \n{RED_START}Claude chat completion usage: \n {token_usage}
                        \nChat completion Total cost (USD): {total_cost}${RED_END}
                        """
            )

            if functions:
                response_message = response.content[0]
                if response_message.function_call:
                    function_name = response_message.function_call.name
                    function_args = response_message.function_call.arguments
                    text = response_message.message.content
                    func_call = (function_name, function_args)
                else:
                    text = response_message.message.content
                    func_call = None
            else:
                text = response.content[0].text
                func_call = None

            queue.put(
                (text, func_call, token_usage)
            )  # Return both text and function call

        except Exception as e:
            logger.error(f"Chat completion error occurred for this prompt: {messages}")
            queue.put((f"An error occurred: {e}", None))

    def chat_complete_claude(self, messages, functions=None):
        queue = Queue()
        claude_process = Process(
            target=self.claude_operation, args=(messages, queue, functions)
        )
        claude_process.start()

        def terminate_process():
            if claude_process.is_alive():
                print(
                    "Timeout occurred for claude chat completion. Terminating process..."
                )
                claude_process.terminate()

        timer = threading.Timer(float(self.config["timeout"]), terminate_process)
        timer.start()

        claude_process.join()

        timer.cancel()

        if not queue.empty():
            (
                message,
                func_call,
                token_usage,
            ) = queue.get()  # Get both message and function call
            return message, func_call, token_usage
        else:
            return "Timeout", None, None  # Return "Timeout" and None for function call


if __name__ == "__main__":
    load_dotenv()
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    config = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 1000,
        "temperature": 0.3,
        "timeout": 300,
    }

    # Messages for the chat conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Can dogs fly?",
        },
    ]

    # Instantiate the class
    completion = AnthropicTextCompletion(config=config, anthropic_api_key=anthropic_api_key)

    # Test the chat completion
    response = completion.chat_complete_claude(messages)

    # Print the response
    print(response)
