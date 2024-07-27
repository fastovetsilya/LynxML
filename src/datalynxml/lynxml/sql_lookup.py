import logging
import csv
import os
import sys
import copy
from dotenv import load_dotenv
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

from datalynxml.lynxml.utils import QueryData
from datalynxml.text.text_vectorization.openai_vectorization import (
    OpenAITextVectorization,
)
from datalynxml.data.database.backend_db_tools import BackendDatabase
from datalynxml.data.database.knowledge_vault_utils import (
    find_adaptive_examples,
)
from datalynxml.data.database.testing.bird_db_tools import TestDatabase
from datalynxml.data.database.integrations.integration import get_backend_db_type
from datalynxml.data.database.integrations.multi_sql import MultiDBIntegration
from datalynxml.data.database.integrations.big_query_sql import BigQueryIntegration
from datalynxml.text.text_completion.openai_completion.gpt_chat_completion import (
    OpenAITextCompletion,
)
from datalynxml.text.text_completion.anthropic_completion.claude_chat_completion import (
    AnthropicTextCompletion,
)
from datalynxml.lynxml.prompts import (
    SQL_LOOKUP_PROMPT_MSGS,
    SQL_LOOKUP_IMPROVEMENT_MSG,
    SQL_QUERY_EXPLAIN_MSGS,
)

from datalynxml.lynxml.gpt_config.gpt_sql_query_config import gpt_sql_query_config
from datalynxml.lynxml.gpt_config.gpt_sql_explanation_config import (
    gpt_sql_explanation_config,
)
from datalynxml.lynxml.anthropic_config.claude_sql_query_config import (
    claude_sql_query_config,
)
from datalynxml.lynxml.anthropic_config.claude_sql_explanation_config import (
    claude_sql_explanation_config,
)
from datalynxml.text.utils import (
    count_tokens,
    truncate_string,
    compute_cosine_similarity,
)

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)

COST_PER_TOKEN_PROMPT = 0.005 / 1000  # 0.01, 0.015
COST_PER_TOKEN_SAMPLED = 0.015 / 1000  # 0.03, 0.075


class SQLLookupPromptMsg(BaseModel_v2):
    role: str
    content: str


class SQLLookupPrimitive(BaseModel_v2):
    """
    A class for performing SQL lookups, generating and explaining SQL queries, and interacting with a backend database.

    Attributes:
        db_info_id (int): Identifier for the database information.
        backend_db_host (str): Host address of the backend database.
        backend_db_password (str): Password for accessing the backend database.
        encryption_key (str): Key used for encryption purposes.
        openai_api_key (str): API key for OpenAI services.
        db_integration_type (Optional[str]): Type of database integration, e.g., PostgreSQL, MySQL, Oracle.
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

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    db_info_id: Optional[int] = None
    chat_id: Optional[int] = None
    backend_db_host: Optional[str] = None
    backend_db_password: Optional[str] = None
    encryption_key: Optional[str] = None
    db_integration_type: Optional[str] = None
    test_db_path: Optional[str] = None

    # Public attributes, but that are not set
    chat_history: Optional[List[SQLLookupPromptMsg]] = Field_v2(default_factory=list)
    backend_database: Optional[BackendDatabase] = Field_v2(default_factory=str)
    test_database: Optional[BackendDatabase] = Field_v2(default_factory=str)
    db_integration: Optional[MultiDBIntegration] = Field_v2(default_factory=str)
    db_company_info: Optional[str] = Field_v2(default_factory=str)
    db_constraints: Optional[str] = Field_v2(default_factory=str)

    # Private attributes, not part of the model schema
    _sql_lookup_prompt: List[SQLLookupPromptMsg] = PrivateAttr_v2()
    _sql_explain_prompt: List[SQLLookupPromptMsg] = PrivateAttr_v2()
    _gpt_chat_completion_query: OpenAITextCompletion = PrivateAttr_v2()
    _claude_chat_completion_query: OpenAITextCompletion = PrivateAttr_v2()
    _gpt_chat_completion_explanation: OpenAITextCompletion = PrivateAttr_v2()
    _openai_text_vectorization: OpenAITextVectorization = PrivateAttr_v2()
    _db_tables_constraints: str = PrivateAttr_v2()

    class Config:
        arbitrary_types_allowed = False

    def __init__(self, **data):
        super().__init__(**data)
        self._sql_lookup_prompt = SQL_LOOKUP_PROMPT_MSGS
        self._sql_explain_prompt = SQL_QUERY_EXPLAIN_MSGS

        if self.openai_api_key:
            self._gpt_chat_completion_query = OpenAITextCompletion(
                config=gpt_sql_query_config,
                openai_api_key=self.openai_api_key,
            )
            self._gpt_chat_completion_explanation = OpenAITextCompletion(
                config=gpt_sql_explanation_config,
                openai_api_key=self.openai_api_key,
            )

        if self.anthropic_api_key:
            self._claude_chat_completion_query = AnthropicTextCompletion(
                config=claude_sql_query_config,
                anthropic_api_key=self.anthropic_api_key,
            )
            self._claude_chat_completion_explanation = AnthropicTextCompletion(
                config=claude_sql_explanation_config,
                anthropic_api_key=self.anthropic_api_key,
            )

        if not self.test_db_path:
            self.backend_database = BackendDatabase(
                host=self.backend_db_host, password=self.backend_db_password
            )

            # Get db type from backend
            db_type = get_backend_db_type(db_info_id=self.db_info_id, 
                                          backend_db_host=self.backend_db_host, 
                                          backend_db_password=self.backend_db_password, )
            
            # Initialize integrations
            if db_type in ["postgresql", "mysql", "oracle"]:
                self.db_integration = MultiDBIntegration(
                    backend_db_host=self.backend_db_host,
                    backend_db_password=self.backend_db_password,
                    db_info_id=self.db_info_id,
                    encryption_key=self.encryption_key,
                )
                self.db_company_info = self.db_integration.company_info
                self.db_constraints = self.backend_database.get_db_constraints(
                    db_info_id=self.db_info_id
                )
                self._openai_text_vectorization = OpenAITextVectorization(
                    openai_api_key=self.openai_api_key
                )
                if db_type == "postgresql":
                    self.db_integration_type = "PostgreSQL"
                elif db_type == "mysql":
                    self.db_integration_type = "MySQL"
                elif db_type == "oracle":
                    self.db_integration_type = "Oracle"

            elif db_type == "bigquery":
                self.db_integration = BigQueryIntegration(
                    db_info_id=self.db_info_id,
                    backend_db_host=self.backend_db_host,
                    backend_db_password=self.backend_db_password,
                    encryption_key=self.encryption_key,
                )
                project_id = self.db_integration.project_id
                dataset_id = self.db_integration.dataset_id
                self.db_integration_type = f"BigQuery SQL (Project ID: {project_id}, Dataset ID: {dataset_id})"

            else: 
                error_message = f"Database type {db_type} not supported, abort."
                logger.error(error_message)
                raise Exception(error_message)
            
        else:
            self.test_database = TestDatabase(db_path=self.test_db_path)
            self.db_company_info = ""
            self.db_constraints = self.test_database.collect_constraints()
            self.db_integration_type = "SQLite"

    def record_token_usage(
        self,
        completion_tokens,
        prompt_tokens,
        total_tokens,
    ):
        if not self.test_db_path:
            cost_completion = completion_tokens * COST_PER_TOKEN_SAMPLED
            cost_prompt = prompt_tokens * COST_PER_TOKEN_PROMPT
            cost_total = cost_completion + cost_prompt

            logger.debug("Recording token usage entry to the database")
            self.backend_database.add_token_usage_entry(
                db_info_id=self.db_info_id,
                chat_id=self.chat_id,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                cost_completion_tokens=cost_completion,
                cost_prompt_tokens=cost_prompt,
                cost_total_tokens=cost_total,
            )
        else:
            logger.debug("Skip recording token usage to the database")

    def get_column_descriptions(self, table_names: List[str]) -> str:
        """
        Retrieves column descriptions for the specified tables from a backend database and formats this information into a CSV format.

        This method queries the backend database for column information of the given table names. It then processes this information, extracting the table name, column name, and column description. Finally, it returns a CSV formatted string containing these details.

        Parameters:
        table_names (List[str]): A list of table names for which column descriptions are required.

        Returns:
        str: A CSV formatted string containing the table name, column name, and column description for each column in the specified tables.
        """
        if not self.test_db_path:
            column_info_csv = self.backend_database.get_column_info_for_tables(
                db_info_id=self.db_info_id, table_names=table_names
            )
            csv_reader = csv.reader(StringIO(column_info_csv))
            header = next(csv_reader)  # Assume the first row is a header
            description_index = header.index("column_description")
            table_name_index = header.index("table_name")
            column_name_index = header.index("column_name")
            output = StringIO()
            csv_writer = csv.writer(output)
            csv_writer.writerow(["table_name", "column_name", "description"])
            for row in csv_reader:
                csv_writer.writerow(
                    [
                        row[table_name_index],
                        row[column_name_index],
                        row[description_index],
                    ]
                )
            formatted_csv = output.getvalue()
        else:
            formatted_csv = self.test_database.get_table_columninfo(
                table_names=table_names
            )
        return formatted_csv

    def remove_column_csv(self, csv_data: str, column_to_remove: str) -> str:
        """
        Removes a specified column from a CSV string.

        This method takes a CSV string and a column name as inputs. It processes the CSV string,
        removing the specified column from it. If the column does not exist in the CSV, the original
        data is returned unchanged.

        Parameters:
        csv_data (str): The CSV data in string format.
        column_to_remove (str): The name of the column to be removed.

        Returns:
        str: The modified CSV data as a string with the specified column removed.
             If the specified column does not exist, the original CSV string is returned.
        """
        # Convert the CSV string to a file-like object
        csv_file = StringIO(csv_data)

        # Read the CSV data
        reader = csv.DictReader(csv_file)

        # Check if the column exists
        if column_to_remove not in reader.fieldnames:
            # If the column does not exist, return the original data
            return csv_data

        # Prepare to write the modified data to a string
        output = StringIO()
        fieldnames = [field for field in reader.fieldnames if field != column_to_remove]
        writer = csv.DictWriter(output, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data rows, excluding the specified column
        for row in reader:
            del row[column_to_remove]
            writer.writerow(row)

        # Get the string from the StringIO object
        return output.getvalue()

    def truncate_table_summaries(
        self,
        table_summaries_csv: str,
        db_constraints_csv: str,
        max_token_limit: int = 15000,
        truncate_step: int = 100,
    ) -> str:
        """
        Truncates table summaries to fit within a specified token limit.

        This method is used to truncate table summary data in CSV format to ensure the total token count
        (including company information and database constraints) does not exceed a specified limit.
        It sequentially removes certain columns and, if necessary, truncates the string to meet the token limit.
        If the combined token count of company information and database constraints already exceeds the limit,
        an exception is raised.

        Parameters:
        table_summaries_csv (str): The table summaries in CSV format.
        max_token_limit (int, optional): The maximum token limit. Default is 5000.
        truncate_step (int, optional): The step size for truncating the string. Default is 100.

        Returns:
        str: The truncated table summaries in CSV format. If truncation is not necessary,
             the original table summaries are returned.
        """
        # First, handle the case when the company summary and constraints are too big
        # TODO: for now, throw an error in this case. Think about how to handle it later
        # TODO: I think a better way is to implement limits on company/table/col descriptions in the frontend/backend
        if (
            count_tokens(self.db_company_info) + count_tokens(db_constraints_csv)
        ) > max_token_limit:
            error_message = (
                "Company description and/or database constraints are too big, abort!"
            )
            logger.error(error_message)
            logger.debug(f"The company description is: {self.db_company_info}")
            logger.debug(
                f"Number of tokens for company description: {count_tokens(self.db_company_info)}"
            )
            logger.debug(f"The constraints are: {db_constraints_csv}")
            logger.debug(
                f"Number of tokens for constraints: {count_tokens(db_constraints_csv)}"
            )
            raise Exception(error_message)

        # If the total number of tokens is not exceeded, return the raw table summaries
        tokens_used = (
            count_tokens(self.db_company_info)
            + count_tokens(db_constraints_csv)
            + count_tokens(table_summaries_csv)
        )
        if tokens_used < max_token_limit:
            logger.debug("Using raw table summaries for SQL query generation.")
            return table_summaries_csv

        # If token limit is exceeded, first remove column description, nonnull_count and unique_count columns
        logger.debug(
            "Truncating column description, nonnull_count and unique_count columns from table summary for SQL query generation."
        )
        table_summaries_truncated = self.remove_column_csv(
            table_summaries_csv, "column_description"
        )
        table_summaries_truncated = self.remove_column_csv(
            table_summaries_truncated, "nonnull_count"
        )
        table_summaries_truncated = self.remove_column_csv(
            table_summaries_truncated, "unique_count"
        )
        tokens_used = (
            count_tokens(self.db_company_info)
            + count_tokens(db_constraints_csv)
            + count_tokens(table_summaries_truncated)
        )
        if tokens_used < max_token_limit:
            return table_summaries_truncated

        # If still exceeds the limit, remove the categories
        logger.debug(
            "Truncating categories column from table summary for SQL query generation."
        )
        table_summaries_truncated = self.remove_column_csv(
            table_summaries_truncated, "categories"
        )
        tokens_used = (
            count_tokens(self.db_company_info)
            + count_tokens(db_constraints_csv)
            + count_tokens(table_summaries_truncated)
        )
        if tokens_used < max_token_limit:
            return table_summaries_truncated

        # If the limit is still exceeded, truncate the string as the last resort
        logger.debug(
            "Truncating the string of table summaries (last resort) for SQL query generation."
        )
        truncate_string_tokens = (
            tokens_used
            - count_tokens(self.db_company_info)
            - count_tokens(db_constraints_csv)
        )
        table_summaries_truncated = truncate_string(
            table_summaries_truncated,
            max_tokens=truncate_string_tokens,
            step=truncate_step,
        )
        table_summaries_truncated = table_summaries_truncated + "[TRUNCATED]"
        return table_summaries_truncated

    def filter_dbconstraints_by_tables(self, table_names: List[str]):
        csv_string = self.db_constraints
        # If no constraints, return as is
        if csv_string == "":
            return csv_string
        # Convert the CSV string into a file-like object
        csv_input = StringIO(csv_string)
        # Create a CSV reader to read the input string
        reader = csv.DictReader(csv_input)
        # Assuming the unwanted duplication in header is a mistake, correct the fieldnames by removing any accidental duplication
        corrected_fieldnames = [
            field
            for i, field in enumerate(reader.fieldnames)
            if field not in reader.fieldnames[:i]
        ]
        # Prepare a StringIO object to hold the filtered CSV
        csv_output = StringIO()
        # Create a CSV writer to write the filtered rows to the output string, using corrected fieldnames
        writer = csv.DictWriter(csv_output, fieldnames=corrected_fieldnames)
        # Write the header to the output CSV
        writer.writeheader()
        # Filter rows where 'table_name' or 'referenced_table_name' matches any of the specified table names
        for row in reader:
            if (
                row["table_name"] in table_names
                or row["referenced_table_name"] in table_names
            ):
                writer.writerow(row)
        # Get the contents of the output CSV string
        output_string = csv_output.getvalue()
        # Close the StringIO objects
        csv_input.close()
        csv_output.close()

        self._db_tables_constraints = output_string
        return output_string

    def generate_query(
        self,
        table_names: List[str],
        question: str,
        query_improvement_message: Optional[str] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generates a SQL query based on the given table names and a question, optionally incorporating a query improvement message.

        This method constructs a SQL lookup query by integrating information about the company, database constraints, table summaries, and the question. If provided, it also includes a query improvement message to refine the query generation process. The method maintains a chat history for generating context-aware queries and computes token counts for input and output queries.

        Parameters:
        table_names (List[str]): A list of table names to be included in the SQL query.
        question (str): The question based on which the SQL query is generated.
        query_improvement_message (Optional[str]): An optional message to improve the query generation process.

        Returns:
        Tuple[str, Dict[str, int]]: A tuple containing the generated SQL query as a string and a dictionary with token counts for 'input' and 'output'.
        """
        if not query_improvement_message:
            company_summary = self.db_company_info

            # Get table summaries
            if not self.test_database:
                table_summaries = self.backend_database.get_column_info_for_tables(
                    db_info_id=self.db_info_id, table_names=table_names
                )
            else:
                table_summaries = self.test_database.get_table_columninfo(
                    table_names=table_names
                )

            # Get db constraints
            db_constraints = self.filter_dbconstraints_by_tables(
                table_names=table_names
            )

            # Dynamically truncate table summaries
            table_summaries = self.truncate_table_summaries(
                table_summaries_csv=table_summaries,
                db_constraints_csv=db_constraints,
            )

            if not self.test_db_path:
                # Find adaptive examples and similarities. Retrieve only one example
                (
                    adaptive_example,
                    adaptive_example_cos_similarity,
                ) = find_adaptive_examples(
                    openai_api_key=self.openai_api_key,
                    backend_db_host=self.backend_db_host,
                    backend_db_password=self.backend_db_password,
                    db_info_id=self.db_info_id,
                    text_input=question,
                    n_examples=1,
                )
                if adaptive_example:
                    adaptive_example = adaptive_example[0]
                    adaptive_example_cos_similarity = adaptive_example_cos_similarity[0]

                    # Define adaptive example based on similarity
                    similarity_threshold = 0.5
                    if adaptive_example and (
                        adaptive_example_cos_similarity > similarity_threshold
                    ):
                        adaptive_example = {
                            "term": adaptive_example["term"],
                            "explanation": adaptive_example["explanation"],
                            "query": adaptive_example["query"],
                        }
                    else:
                        adaptive_example = None
                else:
                    adaptive_example = None
            else:
                adaptive_example = None

            # Parse sql prompt
            sql_lookup_prompt_parsed = copy.deepcopy(self._sql_lookup_prompt)
            sql_lookup_prompt_parsed[0]["content"] = sql_lookup_prompt_parsed[0][
                "content"
            ].format(
                self.db_integration_type,
            )
            sql_lookup_prompt_parsed[1]["content"] = sql_lookup_prompt_parsed[1][
                "content"
            ].format(
                self.db_integration_type,
                company_summary,
                db_constraints,
                table_summaries,
                adaptive_example,
                question,
            )
            self.chat_history = sql_lookup_prompt_parsed
            logger.debug(f"SQL lookup prompt parsed is: {sql_lookup_prompt_parsed}")
            if self.openai_api_key:
                (
                    lookup_query,
                    _,
                    lookup_query_usage,
                ) = self._gpt_chat_completion_query.chat_complete_gpt(
                    messages=sql_lookup_prompt_parsed
                )
            if self.anthropic_api_key:
                (
                    lookup_query,
                    _,
                    lookup_query_usage,
                ) = self._claude_chat_completion_query.chat_complete_claude(
                    messages=sql_lookup_prompt_parsed
                )

            query_msg = {"role": "assistant", "content": lookup_query}
            self.chat_history.append(query_msg)

        else:
            improvement_message = copy.deepcopy(SQL_LOOKUP_IMPROVEMENT_MSG)
            improvement_message = improvement_message.format(
                query_improvement_message, table_names
            )
            improvement_message = {"role": "user", "content": improvement_message}
            self.chat_history.append(improvement_message)
            if self.openai_api_key:
                (
                    lookup_query,
                    _,
                    lookup_query_usage,
                ) = self._gpt_chat_completion_query.chat_complete_gpt(
                    messages=self.chat_history
                )
            if self.anthropic_api_key:
                (
                    lookup_query,
                    _,
                    lookup_query_usage,
                ) = self._claude_chat_completion_query.chat_complete_claude(
                    messages=self.chat_history
                )
            lookup_query_msg = {"role": "assistant", "content": lookup_query}
            self.chat_history.append(lookup_query_msg)

        # Record token usage
        if self.openai_api_key and not self.anthropic_api_key:
            self.record_token_usage(
                completion_tokens=lookup_query_usage.completion_tokens,
                prompt_tokens=lookup_query_usage.prompt_tokens,
                total_tokens=lookup_query_usage.total_tokens,
            )
        if self.anthropic_api_key:
            self.record_token_usage(
                completion_tokens=lookup_query_usage.output_tokens,
                prompt_tokens=lookup_query_usage.input_tokens,
                total_tokens=lookup_query_usage.input_tokens
                + lookup_query_usage.output_tokens,
            )

        # Clean up query from unnecessary BS
        # Check that the query is wrapped and remove wrappers
        if lookup_query.startswith("```sql"):
            lookup_query = lookup_query[6:]
        if lookup_query.endswith("```"):
            lookup_query = lookup_query[:-3]

        return lookup_query

    def explain_query(
        self,
        table_names: List[str],
        lookup_query: str,
        query_result: str,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generates an explanation for a given SQL query based on specific table names.

        This method constructs an explanation for a provided SQL query. It uses company information, table descriptions, and column descriptions to generate a context-aware explanation of the SQL query. The method also calculates token counts for both the input and output of the explanation generation process.

        Parameters:
        table_names (List[str]): A list of table names relevant to the SQL query.
        lookup_query (str): The SQL query for which an explanation is needed.

        Returns:
        Tuple[str, Dict[str, int]]: A tuple containing the generated explanation of the SQL query as a string, and a dictionary with token counts for 'input' and 'output'.
        """
        if not self.test_database:
            company_summary = self.db_company_info
            table_descriptions = self.backend_database.get_tables_with_descriptions(
                db_info_id=self.db_info_id, table_names=table_names
            )
            column_descriptions = self.get_column_descriptions(table_names)

            sql_explain_prompt_parsed = copy.deepcopy(self._sql_explain_prompt)
            sql_explain_prompt_parsed[1]["content"] = sql_explain_prompt_parsed[1][
                "content"
            ].format(
                company_summary,
                table_descriptions,
                column_descriptions,
                lookup_query,
                query_result,
            )
            logging.debug(
                f"Query explanation prompt parsed is: {sql_explain_prompt_parsed}"
            )
            (
                query_explanation,
                _,
                explanation_usage,
            ) = self._gpt_chat_completion_explanation.chat_complete_gpt(
                messages=sql_explain_prompt_parsed
            )

            # Record token usage
            self.record_token_usage(
                completion_tokens=explanation_usage.completion_tokens,
                prompt_tokens=explanation_usage.prompt_tokens,
                total_tokens=explanation_usage.total_tokens,
            )

        else:
            logging.debug("Returning empty explanation for test query")
            query_explanation = ""

        return query_explanation

    def get_query_data(self, lookup_query: str) -> QueryData:
        """
        Executes a given SQL query and retrieves the data along with any errors.

        This method takes an SQL query, modifies it if necessary to include a LIMIT clause if one is not already present,
        executes it using the database integration layer, and retrieves the resulting data. It also checks for any errors
        that might occur during the execution of the query. The method returns both the query data and any errors encountered.

        Parameters:
        lookup_query (str): The SQL query to be executed.

        Returns:
        Tuple[Dict[str, Any], Any]: A tuple containing the data retrieved from executing the query as a dictionary,
        and any error that occurred during execution.
        """

        # Check if the query ends with a semicolon and remove it temporarily.
        ends_with_semicolon = lookup_query.strip().endswith(";")
        if ends_with_semicolon:
            lookup_query = lookup_query.strip().rstrip(";")

        # Check if 'LIMIT' is part of the query, case-insensitive.
        if "LIMIT" not in lookup_query.upper():
            lookup_query += " LIMIT 50000"

        # Append the semicolon back if it was originally there.
        if ends_with_semicolon:
            lookup_query += ";"

        # Execute the query based on the database path.
        if not self.test_db_path:
            query_data = self.db_integration.run_custom_query(lookup_query)
        else:
            query_data = self.test_database.run_custom_query(lookup_query)

        return QueryData(**query_data)

    def autovalidate_query(
        self, table_names: List[str], lookup_query: str, max_attempts: int = 5
    ) -> Tuple[str, Dict[str, any], Dict[str, int]]:
        """
        Auto-validates a given SQL query by ensuring it retrieves valid data. If the initial query is invalid, it tries to regenerate a new query using internal methods, up to a specified number of attempts.

        Parameters:
        table_names (List[str]): A list of table names relevant to the SQL query.
        lookup_query (str): The initial SQL query to validate.
        max_attempts (int, optional): Maximum number of attempts to regenerate the query if the initial one is invalid. Defaults to 5.

        Returns:
        Tuple[str, Dict[str, any], Dict[str, int]]: A tuple containing the validated query as a string, the data retrieved from executing the query, and a dictionary with token counts for 'input' and 'output'.

        Raises:
        ValueError: If a valid query cannot be generated after the maximum number of attempts.
        """
        attempt = 0
        while attempt < max_attempts:
            query_data = self.get_query_data(lookup_query)

            # If the query produces valid data, return immediately
            if query_data.results is not None:
                return lookup_query, query_data

            # If not, try to generate a new query using the current method logic
            # (this assumes the original query has a "question" that can be extracted and used)
            # If you have a method to extract question or relevant information from the original query,
            # it should be used here. Otherwise, this is just an example:
            lookup_query = self.generate_query(
                table_names=table_names, question=None, query_improvement_message=query_data.error
            )
            attempt += 1

        # If max attempts reached and still no valid data
        raise ValueError(
            f"Failed to generate a valid query after {max_attempts} attempts."
        )


if __name__ == "__main__":
    load_dotenv()

    db_id = 30
    chat_id = 143
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    encryption_key = os.getenv("ENCRYPTION_KEY")
    backend_db_host = os.getenv("BACKEND_DB_HOST")
    backend_db_password = os.getenv("BACKEND_DB_PASSWORD")

    logger.debug(f"BACKEND DB HOST: {backend_db_host}")

    # # Run for the integrated db
    # sql_lookup_primitive = SQLLookupPrimitive(
    #     backend_db_host=backend_db_host,
    #     backend_db_password=backend_db_password,
    #     db_info_id=db_id,
    #     chat_id=chat_id,
    #     encryption_key=encryption_key,
    #     openai_api_key=openai_api_key,
    #     anthropic_api_key=anthropic_api_key,
    # )
    # logger.debug("SQL lookup class initialized successfully")
    #
    # logger.debug("Trying an example query...")
    # try:
    #     question = "How many customers do we have?"
    #
    #     # Initial query generation
    #     lookup_query = sql_lookup_primitive.generate_query(
    #         table_names=["customers", "orders"],
    #         question=question,
    #     )
    #     logger.debug(f"Lookup SQL query is: {lookup_query}")
    # except Exception as e:
    #     logger.error(f"Failed to generate an example query with error: {e}")

    # Run in test mode
    test_db_path = "/home/ilya/Desktop/BIRD/dev/dev/dev_databases/dev_databases/california_schools/california_schools.sqlite"
    test_question = "How many students do we have?"

    sql_lookup_primitive = SQLLookupPrimitive(
        # openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        test_db_path=test_db_path,
    )
    lookup_query = sql_lookup_primitive.generate_query(
        table_names=["frpm", "schools"],
        question=test_question,
    )
    logger.debug(f"Lookup SQL query is: {lookup_query}")
