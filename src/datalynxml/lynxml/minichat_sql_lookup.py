import logging
import csv
import os
import sys
import copy
from io import StringIO
from typing import Dict, Tuple, List, Any, Optional
from pydantic.v1 import BaseModel, Field
from pydantic import (
    BaseModel as BaseModel_v2,
    Field as Field_v2,
    PrivateAttr as PrivateAttr_v2,
)

root_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(root_path)

from datalynxml.text.text_completion.openai_completion.gpt_chat_completion import (
    OpenAITextCompletion,
)
from datalynxml.text.utils import count_tokens
from datalynxml.lynxml.prompts import (
    MINICHAT_SQL_LOOKUP_PROMPT_MSGS,
    MINICHAT_SQL_LOOKUP_IMPROVEMENT_MSG,
    MINICHAT_SQL_QUERY_EXPLAIN_MSGS,
)

from datalynxml.lynxml.gpt_config.minichat_gpt_sql_query_config import minichat_gpt_sql_query_config
from datalynxml.lynxml.gpt_config.minichat_gpt_sql_explanation_config import minichat_gpt_sql_explanation_config

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


class SQLLookupPromptMsg(BaseModel_v2):
    role: str
    content: str


class SQLLookupPrimitive(BaseModel_v2):
    """
    A class for performing SQL lookups, generating and explaining SQL queries, and interacting with a backend database.

    Attributes:
        openai_api_key (str): API key for OpenAI services.
        db_type (str): Type of database, e.g., PostgreSQL, MySQL, Oracle.
        db_schema (str): Database schema.
        chat_history (Optional[List[SQLLookupPromptMsg]]): History of chat interactions for SQL lookups.

    Private Attributes:
        _sql_lookup_prompt (List[SQLLookupPromptMsg]): Internal list of SQL lookup prompt messages.
        _sql_explain_prompt (List[SQLLookupPromptMsg]): Internal list of SQL explanation prompt messages.
        _gpt_chat_completion_query (OpenAITextCompletion): GPT model for generating SQL queries.
        _gpt_chat_completion_explanation (OpenAITextCompletion): GPT model for explaining SQL queries.

    Methods:
        backend_database: Returns a BackendDatabase instance configured with host and password.
        db_integration: Returns a MultiDBIntegration instance with database configuration.
        db_constraints: Retrieves database constraints based on db_info_id.
        get_column_descriptions: Fetches and formats column descriptions for specified tables.
        generate_query: Generates a SQL query based on input parameters and GPT completions.
        explain_query: Provides an explanation for a given SQL query using GPT completions.
        get_query_data: Executes a SQL query and returns the result.
        autovalidate_query: Automatically validates and, if necessary, regenerates a SQL query.
    """

    openai_api_key: str
    db_schema: str
    db_constraints: str
    db_type: str

    # Public attributes, but that are not set
    chat_history: Optional[List[SQLLookupPromptMsg]] = Field_v2(default_factory=list)

    # Private attributes, not part of the model schema
    _sql_lookup_prompt: List[SQLLookupPromptMsg] = PrivateAttr_v2()
    _sql_explain_prompt: List[SQLLookupPromptMsg] = PrivateAttr_v2()
    _gpt_chat_completion_query: OpenAITextCompletion = PrivateAttr_v2()
    _gpt_chat_completion_explanation: OpenAITextCompletion = PrivateAttr_v2()

    class Config:
        arbitrary_types_allowed = False

    def __init__(self, **data):
        super().__init__(**data)
        self._sql_lookup_prompt = MINICHAT_SQL_LOOKUP_PROMPT_MSGS
        self._sql_explain_prompt = MINICHAT_SQL_QUERY_EXPLAIN_MSGS
        self._gpt_chat_completion_query = OpenAITextCompletion(
            config=minichat_gpt_sql_query_config,
            openai_api_key=self.openai_api_key,
        )
        self._gpt_chat_completion_explanation = OpenAITextCompletion(
            config=minichat_gpt_sql_explanation_config,
            openai_api_key=self.openai_api_key,
        )
        if self.db_type == "postgresql":
            self.db_type = "PostgreSQL"
        elif self.db_type == "mysql":
            self.db_type = "MySQL"
        elif self.db_type == "oracle":
            self.db_type = "Oracle"

    def generate_query(self, question, query_improvement_message=None):
        # Token counter
        token_count = {
            "input": 0,
            "output": 0,
        }
        if not query_improvement_message:
            sql_lookup_prompt_parsed = copy.deepcopy(self._sql_lookup_prompt)
            sql_lookup_prompt_parsed[0]["content"] = sql_lookup_prompt_parsed[0][
                "content"
            ].format(
                self.db_type,
            )
            sql_lookup_prompt_parsed[1]["content"] = sql_lookup_prompt_parsed[1][
                "content"
            ].format(
                self.db_type,
                self.db_schema,
                self.db_constraints,
                question,
            )
            self.chat_history = sql_lookup_prompt_parsed
            logger.info(f"SQL lookup prompt parsed is: {sql_lookup_prompt_parsed}")
            lookup_query, _, _ = self._gpt_chat_completion_query.chat_complete_gpt(
                messages=sql_lookup_prompt_parsed
            )
            query_msg = {"role": "assistant", "content": lookup_query}
            self.chat_history.append(query_msg)
            token_count["input"] += sum(
                [count_tokens(chat["content"]) for chat in sql_lookup_prompt_parsed]
            )

        else:
            improvement_message = copy.deepcopy(MINICHAT_SQL_LOOKUP_IMPROVEMENT_MSG)
            improvement_message = improvement_message.format(query_improvement_message)
            improvement_message = {"role": "user", "content": improvement_message}
            self.chat_history.append(improvement_message)
            lookup_query, _, _ = self._gpt_chat_completion_query.chat_complete_gpt(
                messages=self.chat_history
            )
            lookup_query_msg = {"role": "assistant", "content": lookup_query}
            self.chat_history.append(lookup_query_msg)
            token_count["input"] += sum(
                [count_tokens(chat["content"]) for chat in self.chat_history]
            )

        token_count["output"] += count_tokens(lookup_query)

        return lookup_query, token_count

    def explain_query(self, lookup_query):
        token_count = {
            "input": 0,
            "output": 0,
        }
        sql_explain_prompt_parsed = copy.deepcopy(self._sql_explain_prompt)
        sql_explain_prompt_parsed[1]["content"] = sql_explain_prompt_parsed[1][
            "content"
        ].format(
            self.db_schema,
            self.db_constraints,
            lookup_query,
        )
        logging.debug(
            f"Query explanation prompt parsed is: {sql_explain_prompt_parsed}"
        )
        query_explanation, _, _ = self._gpt_chat_completion_explanation.chat_complete_gpt(
            messages=sql_explain_prompt_parsed
        )
        token_count["input"] += sum(
            [count_tokens(chat["content"]) for chat in sql_explain_prompt_parsed]
        )
        token_count["output"] += count_tokens(query_explanation)
        return query_explanation, token_count


if __name__ == "__main__":
    sql_lookup_primitive = SQLLookupPrimitive(db_info_id=32)
    # question = "From table data, How long on average does it take a firm user to create his first job?"
    # question = "From table data, What is the percentage of revenue from existing customers?"
    question = "From table data, how many customers are registered in total?"

    # Find the relevant tables
    table_names = sql_lookup_primitive.find_relevant_tables(question=question)
    print(f"Relevant tables are: {table_names}")

    # Initial query generation
    lookup_query = sql_lookup_primitive.generate_query(
        table_names=table_names, question=question
    )
    print(f"Lookup SQL query is: {lookup_query}")

    # Autovalidate the query
    print("Running the query to check that it works...")
    query_data, error = sql_lookup_primitive.get_query_data(lookup_query)
    if not query_data:
        print(error)
        if "timeout" not in error:
            print("First query failed. Autovalidating the query.")
            print(error)
            lookup_query, query_data = sql_lookup_primitive.autovalidate_query(
                table_names=table_names, lookup_query=lookup_query
            )
            print(f"Lookup SQL query is: {lookup_query}")
        else:
            query_data = "Error: query failed to execute. No data collected."
    if not query_data:
        query_data = "Error: query failed to execute. No data collected."

    # Ask the user for feedback
    while True:
        user_response = input("Is this approach OK? (yes/no): ").strip().lower()

        # If the user confirms the query is okay
        if user_response == "yes":
            break
        # If the user wants improvements
        elif user_response == "no":
            improvement_message = input("How can the query be improved?: ").strip()
            table_names = sql_lookup_primitive.find_relevant_tables(
                question=question, tables_improvement_message=improvement_message
            )
            print(f"Relevant tables are: {table_names}")
            lookup_query = sql_lookup_primitive.generate_query(
                table_names=table_names,
                question=question,
                query_improvement_message=improvement_message,
            )
            # Autovalidate the query
            print("Running the query to check that it works...")
            query_data, error = sql_lookup_primitive.get_query_data(lookup_query)
            if not query_data:
                print(error)
                if "timeout" not in error:
                    print("First query failed. Autovalidating the query.")
                    print(error)
                    lookup_query, query_data = sql_lookup_primitive.autovalidate_query(
                        table_names=table_names, lookup_query=lookup_query
                    )
                    print(f"Lookup SQL query is: {lookup_query}")
                else:
                    query_data = "Error: query failed to execute. No data collected."
            if not query_data:
                query_data = "Error: query failed to execute. No data collected."

            print(f"Improved SQL query is: {lookup_query}")

        # If the user gives an unexpected answer
        else:
            print("Please answer with 'yes' or 'no'.")

    print("Final SQL query:", lookup_query)
