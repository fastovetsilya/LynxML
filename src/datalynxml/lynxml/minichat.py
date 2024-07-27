import logging
import os
import sys
import copy
import json
import time
import traceback
import base64
from decimal import Decimal
from datetime import date
from dotenv import load_dotenv
from ast import literal_eval
from typing import Dict, Tuple, List, Any, Optional, Union, Type
from pydantic.v1 import BaseModel, Field
from pydantic import (
    BaseModel as BaseModel_v2,
    Field as Field_v2,
    PrivateAttr as PrivateAttr_v2,
)

import pickle
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.schema.agent import AgentFinish
from langchain.tools import StructuredTool
from langchain.adapters.openai import convert_message_to_dict, convert_dict_to_message

root_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(root_path)

from datalynxml.data.database.backend_db_tools import BackendDatabase
from datalynxml.lynxml.prompts import MINICHAT_INIT_PROMPT, MINICHAT_USER_MESSAGE_PROMPT
from datalynxml.lynxml.minichat_sql_lookup import SQLLookupPrimitive

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)
load_dotenv()

# Define the start and end of the ANSI escape codes for red color
RED_START = "\033[31m"
RED_END = "\033[0m"

"""
SELECT
    t.table_schema,
    t.table_name,
    c.column_name,
    c.data_type,
    c.character_maximum_length,
    c.numeric_precision,
    c.column_default,
    c.is_nullable
FROM
    information_schema.tables t
INNER JOIN
    information_schema.columns c ON c.table_schema = t.table_schema AND c.table_name = t.table_name
WHERE
    t.table_schema NOT IN ('information_schema', 'pg_catalog')
ORDER BY
    t.table_schema,
    t.table_name,
    c.ordinal_position;
"""

"""
SELECT
    tc.table_schema,
    tc.table_name,
    tc.constraint_name,
    tc.constraint_type,
    kcu.column_name,
    ccu.table_schema AS foreign_table_schema,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM
    information_schema.table_constraints AS tc
JOIN
    information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
LEFT JOIN
    information_schema.constraint_column_usage AS ccu
    ON tc.constraint_name = ccu.constraint_name
    AND tc.table_schema = ccu.table_schema
WHERE
    tc.table_schema NOT IN ('information_schema', 'pg_catalog')
ORDER BY
    tc.table_schema,
    tc.table_name,
    tc.constraint_name;
"""


EXAMPLE_DB_SCHEMA = """
"table_schema","table_name","column_name","data_type","character_maximum_length","numeric_precision","column_default","is_nullable"
"public","listings","vehicle_id","integer",NULL,32,NULL,"YES"
"public","listings","price","integer",NULL,32,NULL,"NO"
"public","listings","mileage","integer",NULL,32,NULL,"NO"
"public","listings","region","character varying",20,NULL,NULL,"YES"
"public","regions","vehicle_id","integer",NULL,32,NULL,"YES"
"public","regions","city","character varying",100,NULL,NULL,"NO"
"public","regions","state","character varying",2,NULL,NULL,"NO"
"public","regions","county","character varying",100,NULL,NULL,"YES"
"public","regions","region","character varying",20,NULL,NULL,"YES"
"public","vehicles","vehicle_id","integer",NULL,32,"nextval('vehicles_vehicle_id_seq'::regclass)","NO"
"public","vehicles","make","character varying",50,NULL,NULL,"NO"
"public","vehicles","model","character varying",100,NULL,NULL,"NO"
"public","vehicles","year","smallint",NULL,16,NULL,"NO"
"public","vehicles","vin","character varying",17,NULL,NULL,"NO"
"public","vehicles","created_on","timestamp without time zone",NULL,NULL,"CURRENT_TIMESTAMP","NO"
"""

EXAMPLE_DB_CONSTRAINTS = """
"table_schema","table_name","constraint_name","constraint_type","column_name","foreign_table_schema","foreign_table_name","foreign_column_name"
"public","listings","listings_vehicle_id_fkey","FOREIGN KEY","vehicle_id","public","vehicles","vehicle_id"
"public","regions","regions_vehicle_id_fkey","FOREIGN KEY","vehicle_id","public","vehicles","vehicle_id"
"public","vehicles","vehicles_pkey","PRIMARY KEY","vehicle_id","public","vehicles","vehicle_id"
"public","vehicles","vehicles_vin_key","UNIQUE","vin","public","vehicles","vin"
"""


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, bytearray):
            # Convert bytearray to a base64 encoded string
            return base64.b64encode(obj).decode("utf-8")
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
        question (str): The natural language question from which the SQL query will be generated.
            This question should be formulated clearly to facilitate accurate SQL translation.
        improvement_message (Optional[str]): An optional field that can be used to provide additional
            information or context to refine or improve the generated SQL query. This is particularly
            useful for iterative query refinement.

    Usage:
        The model should be instantiated with a specific question and relevant table names.
        The `improvement_message` can be used in subsequent iterations to refine the query.
    """

    question: str = Field(
        ..., description="The question based on which the SQL query will be generated."
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
        db_type (str): Type of database, e.g., PostgreSQL, MySQL, Oracle.
        db_schema (str): Database schema.
        openai_api_key (str): API key for OpenAI services.
        encryption_key (str): Key used for encryption purposes.
        backend_db_host (str): Host address for the backend database.
        backend_db_password (str): Password for accessing the backend database.

    Private Attributes:
        _backend_database (BackendDatabase): Instance for interacting with the backend database.
        _sql_lookup (SQLLookupPrimitive): Tool for generating and executing SQL queries.
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
    db_schema: str
    db_constraints: str
    db_type: str
    openai_api_key: str
    encryption_key: str
    backend_db_host: str
    backend_db_password: str
    logger_adapter: Union[Type[logging.LoggerAdapter], None]

    # Private attributes, not part of the model schema
    _backend_database: BackendDatabase = PrivateAttr_v2()
    _sql_lookup: SQLLookupPrimitive = PrivateAttr_v2()
    _chat_init_prompt: Any = PrivateAttr_v2()  # This should be str type in the future
    _log: Type[logging.LoggerAdapter] = PrivateAttr_v2()

    class Config:
        arbitrary_types_allowed = False

    def __init__(self, **data):
        super().__init__(**data)
        self._log = (
            self.logger_adapter(logger, {"chat_id": self.chat_id})
            if self.logger_adapter
            else logger
        )
        self._log.debug("Begin initialization")
        init_start_time = time.time()

        self._log.debug("Loading BackendDatabase instance...")
        self._backend_database = BackendDatabase(
            host=self.backend_db_host, password=self.backend_db_password
        )
        self._log.debug("Loading SQLLookupPrimitive instance...")
        self._sql_lookup = SQLLookupPrimitive(
            openai_api_key=self.openai_api_key,
            db_schema=self.db_schema,
            db_constraints=self.db_constraints,
            db_type=self.db_type,
        )
        self._minichat_init_prompt = copy.deepcopy(MINICHAT_INIT_PROMPT)
        self._minichat_user_message_prompt = copy.deepcopy(MINICHAT_USER_MESSAGE_PROMPT)

        self._log.debug("Completed Chat instance init")
        self._log.debug(
            f"{RED_START}Chat initialization took the total of {time.time() - init_start_time} seconds{RED_END}"
        )

    def ask_database(
        self,
        question: str,
        improvement_message: Optional[str] = None,
    ) -> str:
        """
        Generates, runs, and explains an SQL query based on a given question and table names, then returns the results in JSON format.
        """

        self._log.debug("Start ask_database call")
        method_start_time = time.time()

        if not improvement_message:
            lookup_query, sql_token_count = self._sql_lookup.generate_query(
                question=question
            )
        else:
            lookup_query, sql_token_count = self._sql_lookup.generate_query(
                question=question,
                query_improvement_message=improvement_message,
            )

        self._log.debug(f"Lookup SQL query is: {lookup_query}")

        # Step 2: Explain the query
        # TODO: re-enable if need to explain the query
        # query_explanation, explanation_token_count = self._sql_lookup.explain_query(
        #     lookup_query=lookup_query
        # )
        # text = "\n\nQuery Explanation: " + query_explanation
        text = ""  # Empty string instead of query explanation

        response = {"text": text, "query": lookup_query}
        response = json.dumps(response, cls=CustomEncoder)

        self._log.debug(
            f"{RED_START}ask_database took {time.time() - method_start_time} seconds{RED_END}"
        )

        return response

    def load_chat_memory(self):
        # Initialize memory with the chat history
        memory = ConversationSummaryBufferMemory(
            llm=ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=self.openai_api_key,
            ),
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2048,
        )

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

    def save_chat_memory(self, memory):
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

    def is_jsonified_dict(self, input_str):
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

    def get_next_response(self, user_message):
        self._log.debug("Start get_next_response call")
        method_start_time = time.time()

        # Initialize new chat meta function
        def init_new_chat_meta():
            """
            general_response: regular response with a chat
            """
            new_chat_meta = {
                "tables_lookup_history": None,
                "sql_lookup_history": None,
            }
            return new_chat_meta

        # Check if the chat meta exists
        self._log.debug("Loading chat meta...")
        chat_meta = self._backend_database.get_chat_meta(self.chat_id)
        if not chat_meta:
            chat_meta = init_new_chat_meta()
        # self._log.debug(f"Chat meta is: {chat_meta}")

        # Regular chat state
        self._log.debug('Executing upon chat_meta as "general_response"')

        # Load table and sql lookups history
        self._sql_lookup.chat_history = chat_meta["sql_lookup_history"]

        # Initialize the LLM
        llm = ChatOpenAI(
            temperature=0.1,
            model_name="gpt-4-1106-preview",
            model_kwargs={"presence_penalty": 1.5},
            openai_api_key=self.openai_api_key,
        )
        tools = [
            StructuredTool.from_function(
                func=self.ask_database,
                args_schema=AskDatabaseInput,
                description="Function to ask database",
            ),
        ]

        llm_with_tools = llm.bind(
            functions=[format_tool_to_openai_function(t) for t in tools]
        )

        # Using LCEL to create the agent
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{self._minichat_init_prompt}",
                ),
                (
                    "user",
                    self._minichat_user_message_prompt.format(
                        "{chat_history}", "{input}"
                    ),
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )

        def parse(output):
            # Check if a function was invoked in the output
            if "function_call" in output.additional_kwargs:
                # Parse out the function call
                function_call = output.additional_kwargs["function_call"]
                name = function_call["name"]
                inputs = json.loads(function_call["arguments"])

                # If the function was 'ask_database', execute it and return the result
                if name == "ask_database":
                    # Assuming that 'ask_database' function is defined and can be called here
                    # You might need to adjust this part based on how your 'ask_database' function is implemented
                    result = self.ask_database(**inputs)

                    # Save table and sql lookups history
                    chat_meta["sql_lookup_history"] = self._sql_lookup.chat_history
                    self._backend_database.add_update_chat_meta(
                        chat_id=self.chat_id, chat_meta=chat_meta
                    )

                    return AgentFinish(
                        return_values={"output": result}, log=str(result)
                    )
                else:
                    # Handle other functions if necessary
                    pass
            else:
                # If no function was invoked, return the original output to the user
                return AgentFinish(
                    return_values={"output": output.content}, log=output.content
                )

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | llm_with_tools
            | parse
        )

        # Load chat memory
        memory = self.load_chat_memory()

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
        )

        chat_history = memory.load_memory_variables({}).get("chat_history")
        print(f"Chat history: {chat_history}")
        response = agent_executor.invoke(
            {"input": user_message, "chat_history": chat_history}
        )
        response = response.get("output")
        print(f"Response: {response}")

        # Save chat memory
        self.save_chat_memory(memory)

        if not self.is_jsonified_dict(response):
            response = {"text": str(response)}
            response = json.dumps(response, cls=CustomEncoder)

        # Check that the [ERROR] identifier is present
        response_dict = json.loads(response)
        if "query" in response_dict:
            if "[ERROR]" in response_dict["query"]:
                response = {"text": response_dict["query"].replace("[ERROR]", "")}
                response = json.dumps(response, cls=CustomEncoder)

        self._log.debug(f"Return response dictionary is: {response}")
        self._log.debug(
            f"{RED_START}get_next_response took {time.time() - method_start_time} seconds{RED_END}"
        )

        return response


if __name__ == "__main__":
    chat_id = "40"  # replace with actual chat id

    openai_api_key = os.getenv("OPENAI_API_KEY")
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

    while True:
        user_message = input("You: ")
        chat_instance = Chat(
            chat_id=chat_id,
            db_type="mysql",
            db_schema=EXAMPLE_DB_SCHEMA,
            db_constraints=EXAMPLE_DB_CONSTRAINTS,
            openai_api_key=openai_api_key,
            encryption_key=encryption_key,
            backend_db_host=backend_db_host,
            backend_db_password=backend_db_password,
            logger_adapter=None,
        )
        # Add user message to chat_history
        assistant_response = chat_instance.get_next_response(user_message=user_message)
        # Add assistant message to chat_history
        response_message = json.loads(assistant_response)["text"]
        if response_message:
            print("Assistant: ", assistant_response)
