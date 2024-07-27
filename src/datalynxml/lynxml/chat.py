import logging
import pickle
import os
import re
import sys
import copy
import json
import time
import base64
from typing import Generator, Type
from decimal import Decimal
from datetime import date
from dotenv import load_dotenv
from uuid import UUID
from typing import Dict, Tuple, List, Union, Any, Optional, Type
from json import JSONDecodeError
from pydantic.v1 import BaseModel, Field
from pydantic import (
    BaseModel as BaseModel_v2,
    Field as Field_v2,
    PrivateAttr as PrivateAttr_v2,
)

from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.agents import AgentActionMessageLog
from langchain_core.exceptions import OutputParserException
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.messages.human import HumanMessage
from langchain.agents import (
    LLMSingleActionAgent,
    Tool,
    AgentExecutor,
    AgentOutputParser,
    AgentType,
    initialize_agent,
    load_tools,
)
from langchain.prompts import (
    ChatPromptTemplate,
    StringPromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.tools import StructuredTool
from langchain.adapters.openai import convert_message_to_dict, convert_dict_to_message

root_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(root_path)

from datalynxml.data.database.backend_db_tools import BackendDatabase
from datalynxml.data.database.knowledge_vault_utils import (
    get_rules,
)
from datalynxml.data.database.testing.bird_db_tools import TestDatabase
from datalynxml.lynxml.prompts import (
    CHAT_INIT_PROMPT,
    CHAT_POST_INIT_PROMPT,
    CHAT_USER_MESSAGE_PROMPT,
)
from datalynxml.lynxml.sql_lookup import SQLLookupPrimitive
from datalynxml.text.text_completion.openai_completion.gpt_chat_completion import (
    OpenAITextCompletion,
)
from datalynxml.lynxml.gpt_config.gpt_header_config import (
    gpt_header_config,
)
from datalynxml.lynxml.prompts import CHAT_HEADER_MSGS
from datalynxml.lynxml.utils import QueryData, QueryDataTruncated, intelligent_truncate_data, ModelResponse
from datalynxml.text.utils import count_tokens, truncate_string


logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)

# Define the start and end of the ANSI escape codes for red color
RED_START = "\033[31m"
RED_END = "\033[0m"

BLUE_START = "\033[94m"
BLUE_END = "\033[0m"

GREEN_START = "\033[32m"
GREEN_END = "\033[0m"

COST_PER_TOKEN_PROMPT = 0.005 / 1000  # 0.01, 0.03
COST_PER_TOKEN_SAMPLED = 0.015 / 1000  # 0.03, 0.06


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


class AskDatabaseInput(BaseModel):
    """
    Represents request parameters for generating SQL queries based on user-provided questions and table names.

    This model is designed to encapsulate the necessary information for constructing SQL queries
    from natural language questions. It ensures that the required elements for query generation,
    such as table names and the question itself, are provided and adhere to specified constraints.

    Attributes:
        name (str): The name of the query. Used to display it to the user as a header.
        question (str): The natural language question from which the SQL query will be generated.
            This question should be formulated clearly to facilitate accurate SQL translation.
        table_names (List[str]): A list of table names that are relevant to the question. These
            table names are used as a reference for generating the SQL query. The list must contain
            a minimum of two and a maximum of five table names to maintain query complexity and performance.
        improvement_message (Optional[str]): An optional field that can be used to provide additional
            information or context to refine or improve the generated SQL query. This is particularly
            useful for iterative query refinement.

    Usage:
        The model should be instantiated with a specific question and relevant table names.
        The `improvement_message` can be used in subsequent iterations to refine the query.
    """

    name: str = Field(
        ...,
        description="""
        The name of the query. 
        Used to display it to the user as a header. Not used in query generation.
        """,
    )
    question: str = Field(
        ...,
        description="The question based on which the SQL query will be generated.",
    )
    table_names: List[str] = Field(
        ...,
        description="""
        A list of table names to be used in generating the SQL query. 
        Always use at least 5 tables, and maximum 10 tables.
        """,
        min_items=5,  # Minimum number of table names required
        max_items=10,  # Maximum number of table names allowed
    )
    improvement_message: Optional[str] = Field(
        None,
        description="""
        A message with detailed instructions on how to modify the query. 
        Can be used to create the next query based on the previous query.
        Modifies the latest generated query.
        """,
    )


class Chat(BaseModel_v2):
    """
    A class representing a chat interface that integrates various functionalities
    including database interactions, chat history management, and response generation
    using language models.

    Attributes:
        chat_id (str): Unique identifier for the chat session.
        db_id (Optional[int]): Identifier for the database, if applicable.
        openai_api_key (str): API key for OpenAI services.
        encryption_key (str): Key used for encryption purposes.
        backend_db_host (str): Host address for the backend database.
        backend_db_password (str): Password for accessing the backend database.
        sql_lookup (SQLLookupPrimitive): Tool for generating and executing SQL queries.
        logger_adapter: (logging.LoggerAdapter): Logging adapter to keep logs context consistent throughout the app
        test_configs (Optional[Dict]): Configurations for the tests. None by default


    Private Attributes:
        _backend_database (BackendDatabase): Instance for interacting with the backend database.
        _chat_init_prompt (Any): Initial prompt for the chat, to be formatted with dynamic content.

    Methods:
        __init__(**data): Constructor for the Chat class.
        parse_chat_init_prompt(): Parses and formats the initial chat prompt.
        ask_database(question, table_names, improvement_message): Generates and executes SQL queries based on user input.
        load_chat_memory(): Loads chat memory for the current session.
        save_chat_memory(memory): Saves the current chat memory.
        is_jsonified_dict(input_str): Checks if a string is a valid JSON-formatted dictionary.
        get_next_response(): Generates the next response in the chat based on user input and context.
    """

    chat_id: str
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    db_id: Optional[int] = None
    encryption_key: Optional[str] = None
    backend_db_host: Optional[str] = None
    backend_db_password: Optional[str] = None
    sql_lookup: SQLLookupPrimitive = None
    logger_adapter: Union[Type[logging.LoggerAdapter], None]
    test_configs: Optional[Dict] = None

    # Private attributes, not part of the model schema
    _backend_database: BackendDatabase = PrivateAttr_v2()
    _gpt_chat_completion: OpenAITextCompletion = PrivateAttr_v2()
    _chat_init_prompt: str = PrivateAttr_v2()
    _chat_post_init_prompt: str = PrivateAttr_v2()
    _chat_user_message_prompt: str = PrivateAttr_v2()
    _agent: Any = PrivateAttr_v2()  # Wtf is its type?!
    _log: Type[logging.LoggerAdapter] = PrivateAttr_v2()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._log = (
            self.logger_adapter(logger, {"chat_id": self.chat_id})
            if self.logger_adapter
            else logger
        )
        self._log.debug("Begin initialization")
        init_start_time = time.time()

        if not self.test_configs:
            self._log.debug("Loading BackendDatabase instance...")
            self._backend_database = BackendDatabase(
                host=self.backend_db_host, password=self.backend_db_password
            )

            # Initialize gpt chat completion
            self._gpt_chat_completion = OpenAITextCompletion(
                config=gpt_header_config,
                openai_api_key=self.openai_api_key,
            )

            # Initialize SQL lookup generator
            self._log.debug("Loading SQLLookupPrimitive instance...")
            if not self.anthropic_api_key:
                self.sql_lookup = (
                    SQLLookupPrimitive(
                        backend_db_host=self.backend_db_host,
                        backend_db_password=self.backend_db_password,
                        db_info_id=self.db_id,
                        chat_id=self.chat_id,
                        encryption_key=self.encryption_key,
                        openai_api_key=self.openai_api_key,
                    )
                    if self.db_id
                    else None
                )
            if self.anthropic_api_key:
                self.sql_lookup = (
                    SQLLookupPrimitive(
                        backend_db_host=self.backend_db_host,
                        backend_db_password=self.backend_db_password,
                        db_info_id=self.db_id,
                        chat_id=self.chat_id,
                        encryption_key=self.encryption_key,
                        openai_api_key=self.openai_api_key,
                        anthropic_api_key=self.anthropic_api_key,
                    )
                    if self.db_id
                    else None
                )
        else:
            if not self.anthropic_api_key:
                self.sql_lookup = SQLLookupPrimitive(
                    openai_api_key=self.openai_api_key,
                    test_db_path=self.test_configs["db_path"],
                )
            if self.anthropic_api_key:
                self.sql_lookup = SQLLookupPrimitive(
                    anthropic_api_key=self.anthropic_api_key,
                    test_db_path=self.test_configs["db_path"],
                )

        self._chat_init_prompt = copy.deepcopy(CHAT_INIT_PROMPT)
        self._chat_post_init_prompt = copy.deepcopy(CHAT_POST_INIT_PROMPT)
        self._log.debug("Parsing chat init prompt...")
        self.parse_chat_init_prompt()
        self._chat_user_message_prompt = copy.deepcopy(CHAT_USER_MESSAGE_PROMPT)
        self._agent = None
        self._log.debug("Initializing agent...")
        self.initialize_agent()

        self._log.debug("Completed Chat instance init")
        self._log.debug(
            f"{RED_START}Chat initialization took the total of {time.time() - init_start_time} seconds{RED_END}"
        )

    def record_token_usage(
        self,
        completion_tokens,
        prompt_tokens,
        total_tokens,
    ):
        if not self.test_configs:
            cost_completion = completion_tokens * COST_PER_TOKEN_SAMPLED
            cost_prompt = prompt_tokens * COST_PER_TOKEN_PROMPT
            cost_total = cost_completion + cost_prompt

            self._log.debug("Recording token usage entry to the database")
            self._backend_database.add_token_usage_entry(
                db_info_id=self.db_id,
                chat_id=self.chat_id,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                cost_completion_tokens=cost_completion,
                cost_prompt_tokens=cost_prompt,
                cost_total_tokens=cost_total,
            )
        else:
            self._log.debug("Skip recording token usage to the database")

    def initialize_agent(self):
        # Initialize the LLM
        llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-4o",
            openai_api_key=self.openai_api_key,
        )
        tools = [
            StructuredTool.from_function(
                func=self.ask_database,
                args_schema=AskDatabaseInput,
                description="Function to ask database",
            )
        ]
        llm_with_tools = llm.bind_tools(tools)

        # Define the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._chat_init_prompt),
                ("system", self._chat_post_init_prompt),
                (
                    "user",
                    self._chat_user_message_prompt.format("{chat_history}", "{input}"),
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )

        # Define agent
        self._agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"],
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )

    def parse_chat_init_prompt(
        self, max_token_limit: int = 10000, truncate_step: int = 50
    ) -> None:
        """
        Constructs an initialization prompt for a chat system using descriptions fetched from a database.

        This function retrieves company information, database constraints, table descriptions, and column descriptions from a database. It then formats these pieces of information into a prompt for initializing a chat session. The text is truncated to adhere to a specified token limit.

        Args:
            max_token_limit (int): The maximum number of tokens allowed in the final prompt. Defaults to 1024.
            truncate_step (int): The number of tokens by which to reduce the text when truncating. Defaults to 50.

        Returns:
            None

        Raises:
            Exception: Logs an error message if an exception occurs during the prompt construction process.
        """
        try:
            # Initialize total token count
            tokens_used = 0

            # Get descriptions from sql lookup primitive
            db_constraints = self.sql_lookup.db_constraints

            if not self.test_configs:
                company_summary = self.sql_lookup.db_integration.company_info
                all_table_descriptions = (
                    self.sql_lookup.backend_database.get_tables_with_descriptions(
                        db_info_id=self.sql_lookup.db_info_id
                    )
                )
                all_enabled_tables = (
                    self.sql_lookup.db_integration.get_all_table_names()
                )
                all_column_descriptions = self.sql_lookup.get_column_descriptions(
                    table_names=all_enabled_tables
                )
            else:
                company_summary = self.sql_lookup.db_company_info
                all_table_descriptions = (
                    self.sql_lookup.test_database.get_tables_info()
                )
                all_column_descriptions = (
                    self.sql_lookup.test_database.get_columns_info()
                )

            # Parse iteratively
            tokens_used += count_tokens(company_summary)
            if tokens_used >= max_token_limit:
                self._log.debug(
                    f"Truncating company summary to {max_token_limit} tokens..."
                )
                company_summary = (
                    truncate_string(
                        company_summary, max_tokens=max_token_limit, step=truncate_step
                    )
                    + "[TRUNCATED]"
                )
                all_table_descriptions = "[TRUNCATED]"
                all_column_descriptions = "[TRUNCATED]"
                db_constraints = "[TRUNCATED]"
            else:
                excess_tokens = tokens_used
                tokens_used += count_tokens(all_table_descriptions)
                if tokens_used >= max_token_limit:
                    self._log.debug(
                        f"Truncating table descriptions to {max_token_limit - excess_tokens} tokens..."
                    )
                    all_table_descriptions = (
                        truncate_string(
                            all_table_descriptions,
                            max_tokens=max_token_limit - excess_tokens,
                            step=truncate_step,
                        )
                        + "[TRUNCATED]"
                    )
                    all_column_descriptions = "[TRUNCATED]"
                    db_constraints = "[TRUNCATED]"
                else:
                    excess_tokens += tokens_used
                    tokens_used += count_tokens(all_column_descriptions)
                    if tokens_used >= max_token_limit:
                        self._log.debug(
                            f"Truncating column descriptions to {max_token_limit - excess_tokens} tokens..."
                        )
                        all_column_descriptions = (
                            truncate_string(
                                all_column_descriptions,
                                max_tokens=max_token_limit - excess_tokens,
                                step=truncate_step,
                            )
                            + "[TRUNCATED]"
                        )
                        db_constraints = "[TRUNCATED]"
                    else:
                        excess_tokens += tokens_used
                        tokens_used += count_tokens(db_constraints)
                        if tokens_used >= max_token_limit:
                            self._log.debug(
                                f"Truncating db constraints to {max_token_limit - excess_tokens} tokens..."
                            )
                            db_constraints = (
                                truncate_string(
                                    db_constraints,
                                    max_tokens=max_token_limit - excess_tokens,
                                    step=truncate_step,
                                )
                                + "[TRUNCATED]"
                            )

            # Get the rules for this database
            if self.db_id:
                logger.debug(f"Chat retrieving the rules for db_info_id: {self.db_id}")
                rules = get_rules(
                    backend_db_host=self.backend_db_host,
                    backend_db_password=self.backend_db_password,
                    db_info_id=self.db_id,
                )
                rules = [r["rule"] for r in rules]
            else:
                logger.warning(
                    "Chat slip retrieving the rules because db_info_id not provided"
                )
                rules = None

            self._chat_init_prompt = self._chat_init_prompt.format(
                rules,
                company_summary,
                all_table_descriptions,
                all_column_descriptions,
                db_constraints,
            )
            # self._log.debug(f"Parsed init prompt is: {self._chat_init_prompt}")
        except Exception:
            self._log.exception(f"An error occurred while parsing chat init prompt")

    def ask_database(
        self,
        name: str,
        question: str,
        table_names: List[str],
        improvement_message: Optional[str] = None,
    ) -> ModelResponse:
        """
        Generates, runs, and explains an SQL query based on a given question and table names, then returns the results in JSON format.
        """

        self._log.debug("Start ask_database call")
        method_start_time = time.time()

        # Step 1: Generate and run the query
        if not self.db_id and not self.test_configs:
            text = "No database available to generate query for. No query generated and no data returned."
            self._log.error(text)
            return ModelResponse(text=text)

        if not improvement_message:
            lookup_query = self.sql_lookup.generate_query(
                table_names=table_names, question=question
            )
        else:
            logger.info(
                f"{GREEN_START}Querying database with improvement message: {improvement_message}{GREEN_END}"
            )
            lookup_query = self.sql_lookup.generate_query(
                table_names=table_names,
                question=question,
                query_improvement_message=improvement_message,
            )

        self._log.debug(f"Lookup SQL query is: {lookup_query}")

        query_data: QueryData = self.sql_lookup.get_query_data(lookup_query)

        if not query_data.results:
            if not query_data.error:
                text = "Query executed successfully but returned no data"
                self._log.debug(text)
            elif "timeout" in query_data.error:
                text = "Query failed to generate due to timeout"
                self._log.error(text)
            else:
                self._log.debug(
                    f"The query failed with this error: {query_data.error}. Autovalidating."
                )
                try:
                    (lookup_query, query_data) = self.sql_lookup.autovalidate_query(
                        table_names=table_names, lookup_query=lookup_query)
                    text = f"Query executed successfully after autovalidation"
                    self._log.debug(text)
                except:
                    error = query_data.error
                    text = f"Tried to autovalidate the query {lookup_query}, but failed with error: {error}. No data returned"
                    self._log.error(text)
                    return ModelResponse(text=text)
        else:
            text = "Query executed successfully"
            self._log.debug(text)

        # Additional check
        if not query_data:
            text = "Query failed to generate/return data for unknown reason. No data returned"
            self._log.error(text)
            return ModelResponse(text=text)

        # Step 2: Explain the query
        self._log.debug("Performing intelligent data truncation...")
        query_data_truncated = QueryDataTruncated(
            column_names=query_data.column_names,
            results=intelligent_truncate_data(
                query_data.results, truncate_max_res=3,
            ),
            error=query_data.error,
        )

        self._log.debug("Query data truncated successfully.")
        self._log.debug(f"{BLUE_START}Query data truncated is: {query_data_truncated}{BLUE_END}")

        query_explanation = self.sql_lookup.explain_query(
            table_names=table_names,
            lookup_query=lookup_query,
            query_result=query_data_truncated,
        )
        explanation_prefix = "Please review the steps to make sure the query is correct:"
        query_explanation = "### " + name + "\n" + explanation_prefix + "\n" + query_explanation

        self._log.debug(
            f"{RED_START}ask_database took {time.time() - method_start_time} seconds{RED_END}"
        )

        return ModelResponse(
            text=query_explanation,
            query=lookup_query,
            query_data=query_data,
            query_data_truncated=query_data_truncated,
        )

    def load_chat_memory(self):
        """
        Loads the chat memory for the current session from the backend database.

        This method initializes a ConversationSummaryBufferMemory instance and retrieves the chat history
        and summary from the backend database using the current chat session's identifier (chat_id). It then
        processes this information, converting JSON data to dictionary format and subsequently to message objects,
        which are then assigned to the ConversationSummaryBufferMemory instance.

        Returns:
            ConversationSummaryBufferMemory: An object containing the loaded chat memory, including both the detailed
                                             chat messages and a summary of the conversation.

        Exceptions:
            - This method might raise exceptions related to database access or data conversion (e.g., JSON parsing errors).
              These exceptions should be handled appropriately where this method is called.

        Side Effects:
            - Retrieves chat memory data from the backend database and processes it for use in the current chat session.
        """
        # Initialize memory with the chat history
        memory = ConversationSummaryBufferMemory(
            llm=ChatOpenAI(
                model_name="gpt-4-turbo",
                openai_api_key=self.openai_api_key,
            ),
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2048,
        )

        if not self.test_configs:
            # Download chat memory
            (
                chat_memory_msgs_json,
                chat_memory_summary,
            ) = self._backend_database.get_langchain_chat_memory(chat_id=self.chat_id)

            # Convert json to dict
            if chat_memory_msgs_json:
                extracted_messages_dict = json.loads(chat_memory_msgs_json)
                # Convert dict to messages
                extracted_messages = [
                    convert_dict_to_message(m) for m in extracted_messages_dict
                ]
            else:
                extracted_messages = []

            memory.chat_memory.messages = extracted_messages
            if chat_memory_summary:
                memory.moving_summary_buffer = chat_memory_summary

        return memory

    def save_chat_memory(self, memory: ConversationSummaryBufferMemory) -> None:
        """
        Saves the current chat memory to the backend database.

        This method extracts messages and summaries from the provided ConversationSummaryBufferMemory instance,
        converts them to a dictionary format, and then serializes them to JSON. It then uploads this serialized
        chat memory along with the summaries to the backend database using the chat session's identifier (chat_id).

        Args:
            memory (ConversationSummaryBufferMemory): The chat memory object containing the current chat session's
                                                      messages and summary to be saved.

        Exceptions:
            - This method might raise exceptions related to data conversion (e.g., JSON serialization errors) or
              database access issues. These exceptions should be handled appropriately where this method is called.

        Side Effects:
            - Converts and uploads the current chat session's memory to the backend database for persistence.
        """

        if not self.test_configs:
            # Extract the messages
            extracted_messages = memory.chat_memory.messages
            extracted_summary = memory.moving_summary_buffer
            # Convert to dict
            extracted_messages_dict = [
                convert_message_to_dict(m) for m in extracted_messages
            ]
            # Jsonify
            extracted_messages_json = json.dumps(extracted_messages_dict)

            # Upload chat memory
            self._backend_database.upsert_langchain_chat_memory_msgs(
                chat_id=self.chat_id,
                chat_memory_msgs_json=extracted_messages_json,
                chat_memory_summary=extracted_summary,
            )

    def is_jsonified_dict(self, input_str: str) -> bool:
        """
        Checks if a given string is a valid JSON-formatted dictionary.

        This method attempts to parse the input string as JSON and then checks if the resulting
        object is a dictionary. It's used to validate whether strings contain JSON representations
        of dictionaries, which is a common data format for exchanging information.

        Args:
            input_str (str): The string to be checked for JSON dictionary format.

        Returns:
            bool: True if the input string is a valid JSON-formatted dictionary, False otherwise.

        Exceptions:
            - Handles json.JSONDecodeError to catch cases where the input string is not valid JSON.

        Side Effects:
            - No significant side effects. The method operates purely as a check and does not modify state.
        """
        try:
            # Attempt to parse the string as JSON
            parsed = json.loads(input_str)

            # Check if the parsed result is a dictionary
            if isinstance(parsed, dict):
                return True
            else:
                return False
        except json.JSONDecodeError:
            # The input is not valid JSON
            return False

    def create_chat_header(self, n_words: int = 3) -> str:
        """
        Generates a chat header based on the chat history.

        This method loads the chat memory to retrieve the chat history, and then generates
        a chat header. The header is created by formatting predefined messages with the
        specified number of words and the actual chat history. The generated header is
        intended to provide a context or introduction for the chat based on its history.

        Args:
        - n_words (int, optional): The number of words to be included in the header from the chat history.
          Default is 4.

        Returns:
        - str: The generated chat header as a string.

        Note:
        This method uses internal logging to debug the process of loading chat memory,
        generating the chat header, and the final chat header itself.
        """
        # Load chat memory
        self._log.debug("Loading chat memory...")
        memory = self.load_chat_memory()
        chat_history = memory.load_memory_variables({}).get("chat_history")
        self._log.debug("Generating chat header...")
        chat_header_prompt = copy.deepcopy(CHAT_HEADER_MSGS)
        chat_header_prompt[1]["content"] = chat_header_prompt[1]["content"].format(
            n_words, chat_history
        )
        chat_header, _, _ = self._gpt_chat_completion.chat_complete_gpt(
            messages=chat_header_prompt
        )
        # Remove punctuation and special characters, keeping only words
        chat_header = re.sub(r"[^\w\s]", "", chat_header)
        # Trim chat header to n words
        chat_header_words = chat_header.split()
        chat_header = " ".join(chat_header_words[:n_words])
        self._log.debug(f"Chat header is: `{chat_header}`")
        return chat_header

    def get_next_response(self, user_message: Optional[str] = None) -> Generator[ModelResponse, Any, Any]:
        """
        Generates the next response in the chat based on user input and context.

        This method handles the process of generating a response to a user's message in the chat.
        It involves loading chat memory, initializing language model tools, constructing a chat prompt,
        invoking the language model, and processing the output. The method also handles updating the
        chat memory with the new interaction and saving any changes.

        Args:
            user_message (str): The user's message to which the system will respond.

        Returns:
            NextResponse: A Pydantic model representing the response generated by the system. It can either contain a
                simple text response or a structured response depending on the context.

        Exceptions:
            - This method might raise exceptions related to data processing, language model invocation,
              or database interactions. Such exceptions should be handled where this method is called.

        Side Effects:
            - Updates and saves the chat memory, and potentially modifies the state of the backend database
              with new chat metadata.
            - Invokes language model tools and processes user input, which can lead to external API calls.

        Notes:
            - The implementation involves several steps, including setting up a language model, constructing prompts,
              handling tool invocation, and parsing the model's output, making it a complex and crucial part of the chat system.
        """
        self._log.debug("Start get_next_response call")
        method_start_time = time.time()

        # Initialize new chat meta function
        def init_new_chat_meta():
            """
            Generate new chat metadata
            """
            new_chat_meta = {
                "tables_lookup_history": None,
                "sql_lookup_history": None,
            }
            return new_chat_meta

        # Filter valid human messages function
        def filter_valid_human_history(chat_history):
            chat_human_valid_history = [
                m
                for m in chat_history
                if (isinstance(m, HumanMessage) and m.content != "")
            ]
            return chat_human_valid_history

        # User message is in the test configs for test pipeline
        if self.test_configs and not user_message:
            user_message = """
                           Answer the question below. Do not use clarification. 
                           Do not stop and proceed directly with the answer. 
                           The question is '{}':
                           """
            user_message = user_message.format(self.test_configs["test_question"])

        # Load chat meta (memory for the sql generator)
        if not self.test_configs:
            self._log.debug("Loading chat meta...")
            chat_meta = self._backend_database.get_chat_meta(self.chat_id)
            if not chat_meta:
                chat_meta = init_new_chat_meta()
        else:
            chat_meta = init_new_chat_meta()

        # Load table and sql lookups history
        self.sql_lookup.chat_history = chat_meta["sql_lookup_history"]

        # Load chat memory
        self._log.debug("Loading chat memory...")
        memory = self.load_chat_memory()
        self._log.debug("Starting the agent...")

        # Intermediate steps
        intermediate_steps = []

        while True:
            chat_history = memory.load_memory_variables({}).get("chat_history")
            with get_openai_callback() as cb:
                output = self._agent.invoke(
                    {
                        "input": user_message,
                        "chat_history": chat_history,
                        "intermediate_steps": intermediate_steps,
                    }
                )
                # Record token usage
                self.record_token_usage(
                    completion_tokens=cb.completion_tokens,
                    prompt_tokens=cb.prompt_tokens,
                    total_tokens=cb.total_tokens,
                )
                self._log.debug(
                    f"{RED_START}\nLangchain chat token usage: \n {cb}{RED_END}"
                )
            if isinstance(output, AgentFinish):
                # Case when the tool is not called
                response = ModelResponse(text=output.return_values["output"])
                # Check the chat history for the same previous user message
                human_msgs = filter_valid_human_history(chat_history)
                if human_msgs and human_msgs[-1].content == user_message:
                    memory.save_context(
                        {"input": ""}, {"output": output.return_values["output"]}
                    )
                else:
                    memory.save_context(
                        {"input": user_message},
                        {"output": output.return_values["output"]},
                    )
                break

            else:
                # Extract response and invoking parts of the log
                output = output[0] # We use only one tool, so only one output 
                responded = re.search(r"responded: (.*)", output.log)
                # invoking = re.search(r"(\nInvoking: .*)\n", output.log)

                # If there is a response, yield it
                if responded:
                    text_response = ModelResponse(text=output.message_log[0].content)
                    # Add context to memory
                    # Check the chat history for the same previous user message
                    human_msgs = filter_valid_human_history(chat_history)
                    if human_msgs and human_msgs[-1].content == user_message:
                        memory.save_context({"input": ""}, {"output": memory_context})
                    else:
                        memory.save_context(
                            {"input": user_message}, {"output": text_response.model_dump_json()}
                        )
                    # self._log.debug(f"{BLUE_START}{text_response.mo}{BLUE_END}")
                    yield text_response

                # Run the tool and extract tool observation
                tool = {"ask_database": self.ask_database}[output.tool]
                observation: ModelResponse = tool(**output.tool_input)

                # Save table and sql lookups history
                chat_meta["sql_lookup_history"] = self.sql_lookup.chat_history
                if not self.test_configs:
                    self._backend_database.add_update_chat_meta(
                        chat_id=self.chat_id, chat_meta=chat_meta
                    )

                observation_for_memory = ModelResponse(
                    text=observation.text,
                    query=observation.query,
                    query_data_truncated=observation.query_data_truncated
                )

                # Add context to memory
                memory_context = f"""
                    Invoked function 'ask_database' with parameters: {output.tool_input}.
                    Function responded: {observation_for_memory.model_dump_json()}.
                """
                intermediate_steps.append((output, memory_context))
                # Save memory context
                # Check the chat history for the same previous user message
                human_msgs = filter_valid_human_history(chat_history)
                if human_msgs and human_msgs[-1].content == user_message:
                    memory.save_context({"input": ""}, {"output": memory_context})
                else:
                    memory.save_context(
                        {"input": user_message}, {"output": memory_context}
                    )
                yield observation

        # Save chat memory
        self.save_chat_memory(memory)

        self._log.debug(
            f"{RED_START}get_next_response took {time.time() - method_start_time} seconds{RED_END}"
        )

        yield response


if __name__ == "__main__":
    chat_id = "143"  # replace with actual chat id

    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    # anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    encryption_key = os.getenv("ENCRYPTION_KEY")
    backend_db_host = os.getenv("BACKEND_DB_HOST")
    backend_db_password = os.getenv("BACKEND_DB_PASSWORD")

    # Set the log level to DEBUG to print everything
    logger.setLevel(logging.DEBUG)

    # Create a StreamHandler to print log messages to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    # Integrated DB example
    chat_instance = Chat(
        chat_id=chat_id,
        db_id=1,  # 108
        openai_api_key=openai_api_key,
        # anthropic_api_key=anthropic_api_key,
        encryption_key=encryption_key,
        backend_db_host=backend_db_host,
        backend_db_password=backend_db_password,
        logger_adapter=None,
    )

    while True:
        user_message = input("You: ")

        # Add user message to chat_history
        for chunk in chat_instance.get_next_response(user_message=user_message):
            if chunk:
                logger.debug(f"Chunk from yield: {chunk}")

    # Generate chat header
    # chat_header = chat_instance.create_chat_header()
    # logger.debug(f"Chat header is: `{chat_header}`")

    # # Test example
    # test_configs = {
    #     "db_path": "/home/ilya/Desktop/BIRD/dev/dev/dev_databases/dev_databases/california_schools/california_schools.sqlite",
    #     "test_question": "How many schools do we have?",
    # }
    #
    # chat_instance = Chat(
    #     chat_id=chat_id,
    #     #openai_api_key=openai_api_key,
    #     anthropic_api_key=anthropic_api_key,
    #     logger_adapter=None,
    #     test_configs=test_configs,
    # )
    #
    # generated_queries = []
    # for chunk in chat_instance.get_next_response():
    #     if chunk:
    #         chunk_dict = json.loads(chunk)
    #         if "query" in chunk_dict:
    #             generated_queries_input = {
    #                 "query": chunk_dict["query"],
    #                 "query_data": chunk_dict["query_data"],
    #             }
    #             generated_queries.append(generated_queries_input)
    #         logger.debug(f"Chunk from yield: {chunk}")
    # logger.debug(f"List of queries from the chat is: \n{generated_queries}")
