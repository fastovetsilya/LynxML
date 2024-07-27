import argparse
import json
import os
import sys
import logging

from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import Dict, Tuple, List, Any, Optional
from pydantic import (
    BaseModel as BaseModel_v2,
    Field as Field_v2,
    PrivateAttr as PrivateAttr_v2,
)

root_path = os.path.abspath(os.path.join(__file__, "../../../../../"))
sys.path.append(root_path)

from datalynxml.data.database.integrations.encryption import decrypt
from datalynxml.app.segment import track
from datalynxml.data.database.backend_db_tools import BackendDatabase

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


class BigQueryIntegration(BaseModel_v2):
    db_info_id: int
    backend_db_host: str
    backend_db_password: str
    encryption_key: str
    source_id: Optional[int] = Field_v2(default_factory=int)
    db_type: Optional[str] = Field_v2(default_factory=str)
    db_info: Optional[Dict] = Field_v2(default_factory=dict)
    company_info: Optional[str] = Field_v2(default_factory=str)
    db_schema: Optional[str] = Field_v2(default_factory=str)
    project_id: Optional[str] = Field_v2(default_factory=str)
    dataset_id: Optional[str] = Field_v2(default_factory=str)

    # Private attributes
    _client: bigquery.Client = PrivateAttr_v2()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.db_info = {}
        if self.db_info_id != 0:
            self.db_info = self.get_backend_db_info()
            self.project_id = self.db_info.get("project_id")
            self.dataset_id = self.db_info.get("dataset_id")
            self.db_type = self.db_info.get("db_type", "").lower()
            self.company_info = self.db_info.get("company_description", "")

            db_schema_value = self.db_info.get("db_schema")
            if db_schema_value is None or db_schema_value == "":
                self.db_schema = "public"
            else:
                self.db_schema = db_schema_value

            # Create BigQuery client
            self._client = self._create_client()

    def _create_client(self):
        credentials_str = self.db_info.get("credentials_str")
        credentials_info = json.loads(credentials_str, strict=False)

        credentials = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(credentials=credentials, project=self.project_id)

    def _connect_backend(self):
        try:
            db = BackendDatabase(
                host=self.backend_db_host, password=self.backend_db_password
            )
            conn = db.connect()
        except Exception as e:
            logger.critical("Issue connecting to backend db")
            logger.error(e)
            raise Exception()
        return conn

    def test_connections(self, project_id, dataset_id, credentials_str):
        # Set db_info
        self.db_info = {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "credentials_str": credentials_str,
        }

        try:
            # Create BigQuery client
            credentials_info = json.loads(credentials_str, strict=False)
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client = bigquery.Client(credentials=credentials, project=project_id)

            # Test connection by querying the dataset
            query = f"SELECT 1 FROM `{project_id}.{dataset_id}.__TABLES__` LIMIT 1"
            query_job = client.query(query)
            result = query_job.result()

            # Check if the query executed successfully
            if result.total_rows > 0:
                return True, None
            else:
                return False, "No tables found in the specified dataset"

        except Exception as error:
            return False, str(error)

    def update_status_dbinfo(self, status_code):
        if not self.db_info_id:
            logger.debug("Unable to update db info status: db_info_id is None. Skip.")
            return

        connection = self._connect_backend()
        cursor = connection.cursor()

        try:
            # Query to update the status for the given ID
            query = "UPDATE db_dbinfo SET status = %s WHERE id = %s"
            cursor.execute(query, (status_code, self.db_info_id))

            connection.commit()

        except Exception as e:
            logger.error(f"Error updating status in backend: {e}")
        finally:
            cursor.close()
            connection.close()

    def update_source_progress(self, progress):
        connection = self._connect_backend()
        cursor = connection.cursor()

        try:
            update_query = """
                UPDATE sources_source
                SET progress = %s
                WHERE id = %s
            """
            cursor.execute(update_query, (progress, self.source_id))

            connection.commit()
            logger.info(f"Updating progress to {progress}")
        except Exception as e:
            logger.error(f"Error inserting/updating sources_source: {e}")
        finally:
            cursor.close()
            connection.close()

    def get_backend_db_info(self):
        # Create a dictionary to store the results
        result_dict = {}

        # Connect to the backend (MySQL in this case)
        connection = self._connect_backend()
        cursor = connection.cursor()

        try:
            # Query to fetch all the columns for the given ID
            query = "SELECT * FROM db_dbinfo WHERE id = %s"
            cursor.execute(query, (self.db_info_id,))

            # Fetch the data
            data = cursor.fetchone()

            if data:
                # Get column names from the description
                column_names = [desc[0] for desc in cursor.description]

                # Combine column names and values to create the dictionary
                result_dict = dict(zip(column_names, data))
                # Decrypting db info
                result_dict["project_id"] = decrypt(
                    enc=result_dict["db_username"], key=self.encryption_key
                )
                result_dict["dataset_id"] = decrypt(
                    enc=result_dict["db_name"], key=self.encryption_key
                )
                result_dict["credentials_str"] = decrypt(
                    enc=result_dict["db_password"], key=self.encryption_key
                )
            else:
                logger.debug(f"No data found for ID: {self.db_info_id}")
        except Exception as e:
            logger.error(f"Error fetching data from backend: {e}")
        finally:
            cursor.close()
            connection.close()

        return result_dict

    def get_all_table_names(self):
        tables = self._client.list_tables(self.dataset_id)
        return [table.table_id for table in tables]

    def record_table_names(self, table_names):
        # Connect to backend database
        backend_conn = self._connect_backend()
        backend_cursor = backend_conn.cursor()
        len_tables = len(table_names)
        processed_tables = 0

        # Add table names to db_tableinfo
        for index, table_name in enumerate(table_names):
            # Check if the row already exists
            check_sql = """
            SELECT 1 FROM db_tableinfo
            WHERE table_name = %s AND db_info_id = %s
            """

            try:
                backend_cursor.execute(
                    check_sql,
                    (
                        table_name,
                        self.db_info_id,
                    ),
                )
                exists = backend_cursor.fetchone()

                # If the row doesn't exist, insert it
                if not exists:
                    insert_sql = """INSERT INTO db_tableinfo (table_name, db_info_id) VALUES (%s, %s)"""
                    try:
                        backend_cursor.execute(
                            insert_sql,
                            (
                                table_name,
                                self.db_info_id,
                            ),
                        )
                        backend_conn.commit()
                    except Exception as e:
                        logger.error(
                            f"Error inserting table {table_name} into db_tableinfo: {str(e)}"
                        )
                        self.update_status_dbinfo("E3")
                        backend_conn.rollback()
                        raise Exception()

                # Update source progress
                processed_tables += 1
                progress = int((processed_tables / len_tables) * 10)
                self.update_source_progress(progress)

            except Exception as e:
                logger.exception("Error recording table names")
                self.update_status_dbinfo("E3")
                backend_conn.close()
                raise e

        backend_conn.close()  # Make sure to close the connection

    def get_recorded_table_info(self):
        """
        Retrieves all table names and their respective IDs for the specified db_info_id from the backend MySQL database.

        Returns:
            List[Tuple[int, str]]: A list of tuples where each tuple contains the table ID and its name for the specified db_info_id.
        """
        table_info = []

        # Connect to the backend MySQL database
        conn = self._connect_backend()
        cursor = conn.cursor()

        try:
            # Execute the SELECT statement to get table IDs and names
            query = "SELECT id, table_name FROM db_tableinfo WHERE db_info_id = %s"
            cursor.execute(query, (self.db_info_id,))

            # Fetch all table IDs and names and append them to the list as tuples
            for row in cursor.fetchall():
                table_info.append((row[0], row[1]))

        except Exception as e:
            logger.error(f"Error occurred: {e}")
            self.update_status_dbinfo("E3")
            raise e
        finally:
            # Close the cursor and the connection
            cursor.close()
            conn.close()

        return table_info

    def get_nth_entry(self, table_name, n):
        query = f"SELECT * FROM `{self.project_id}.{self.dataset_id}.{table_name}` LIMIT {n}"
        query_job = self._client.query(query)
        result_dict = {"column_names": None, "results": None, "error": None}

        try:
            results = query_job.result()  # Ensures the query completes

            if query_job.done():
                if query_job.errors:
                    result_dict["error"] = "Query completed with errors: " + str(
                        query_job.errors
                    )
                    return result_dict

                column_names = (
                    [field.name for field in query_job.schema]
                    if query_job.schema
                    else []
                )
                result_rows = [dict(row) for row in results] if results else []

                if not result_rows:
                    result_dict["error"] = "Query completed but returned no rows"
                    return result_dict

                # Assuming 'n' is less than or equal to the number of rows requested in LIMIT
                if len(result_rows) >= n:
                    nth_result = result_rows[n - 1]
                    result_dict["column_names"] = column_names
                    result_dict["results"] = nth_result

                else:
                    result_dict[
                        "error"
                    ] = f"Requested the {n}th entry, but only {len(result_rows)} rows are available."

            else:
                result_dict["error"] = "Query did not complete"

        except TimeoutError as te:
            result_dict["error"] = f"Query exceeded allowed time: {str(te)}"
        except Exception as ex:
            result_dict["error"] = "Error processing query results: " + str(ex)
        return result_dict

    def get_column_data_types(self, table_name):
        table_ref = self._client.dataset(self.dataset_id).table(table_name)
        table = self._client.get_table(table_ref)
        return {field.name: field.field_type for field in table.schema}

    def record_constraints(self):
        constraints = []

        # Ensure the query string is formatted correctly
        query = f"""
        SELECT 
            constraint_catalog AS constraint_schema,
            table_name,
            constraint_name,
            constraint_type,
            NULL AS column_name,
            NULL AS foreign_table_name,
            NULL AS foreign_column_name
        FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS`
        WHERE constraint_name IS NOT NULL
        """

        query_job = self._client.query(query)
        results = query_job.result()

        for row in results:
            constraint = (
                row.constraint_schema,
                row.table_name,
                row.constraint_name,
                row.constraint_type,
                row.column_name,
                row.foreign_table_name,
                row.foreign_column_name,
            )
            constraints.append(constraint)

        # Connecting to the MySQL backend
        mysql_conn = self._connect_backend()
        mysql_cursor = mysql_conn.cursor()

        len_constraints = len(constraints)
        processed_constraints = 0

        logger.debug(f"Total constraints retrieved: {len_constraints}")
        for index, constraint in enumerate(constraints):
            logger.debug(f"Recording constraint: {constraint}")
            # check whether a row with the specified constraint_name and db_info_id already exists
            check_query = """
            SELECT 1 FROM db_constraintsinfo 
            WHERE constraint_name = %s AND db_info_id = %s;
            """
            try:
                mysql_cursor.execute(check_query, (constraint[2], self.db_info_id))
                exists = mysql_cursor.fetchone()

                # If a row exists, perform an UPDATE, otherwise perform an INSERT
                if exists:
                    update_query = """
                    UPDATE db_constraintsinfo
                    SET 
                        column_name = %s,
                        referenced_column_name = %s,
                        referenced_table_name = %s,
                        constraint_type = %s,
                        table_info_id = (
                            SELECT id FROM db_tableinfo WHERE table_name = %s AND db_info_id = %s ORDER BY id DESC LIMIT 1
                        ),
                        table_schema = %s
                    WHERE 
                        constraint_name = %s AND db_info_id = %s
                    """
                    mysql_cursor.execute(
                        update_query,
                        (
                            constraint[4],
                            constraint[6],
                            constraint[5],
                            constraint[3],
                            constraint[1] or constraint[5],  # table_name for sub-query
                            self.db_info_id,  # db_info_id for sub-query
                            None,
                            constraint[2],
                            self.db_info_id,
                        ),
                    )
                else:
                    insert_query = """
                    INSERT INTO db_constraintsinfo (
                        column_name, 
                        constraint_name, 
                        referenced_column_name, 
                        referenced_table_name, 
                        db_info_id, 
                        constraint_type, 
                        table_info_id, 
                        table_schema
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, 
                        (SELECT id FROM db_tableinfo WHERE table_name = %s AND db_info_id = %s ORDER BY id DESC LIMIT 1),
                        %s)
                    """
                    mysql_cursor.execute(
                        insert_query,
                        (
                            constraint[4],
                            constraint[2],
                            constraint[6],
                            constraint[5],
                            self.db_info_id,
                            constraint[3],
                            constraint[1] or constraint[5],  # table_name for sub-query
                            self.db_info_id,  # db_info_id for sub-query
                            None,
                        ),
                    )
                mysql_conn.commit()

                # Update progress, assuming the constraints stage is 10% of the total process and starts at 90%
                processed_constraints += 1
                progress = int(90 + (processed_constraints / len_constraints) * 10)
                self.update_source_progress(progress)

            except Exception as e:
                logger.exception("Error recording constraints")
                self.update_status_dbinfo("E3")
                mysql_conn.close()
                raise e
        mysql_cursor.close()
        mysql_conn.close()

    def record_table_columninfo(
        self,
        n=1,
        max_categories=100,
        max_cat_length=15,
        include_categories=True,
        include_data_type=True,
        include_nonnull_count=True,
        include_unique_count=True,
        include_example_entry=True,
        ignore_description="ignore",
    ):
        
        table_info = self.get_recorded_table_info()

        # Calculate the total number of columns to update progress accurately
        total_columns = 0
        for _, table_name in table_info:
            column_data_types = self.get_column_data_types(table_name)
            total_columns += len(column_data_types)
        processed_columns = 0  # Counter for processed columns to update progress

        for index, (table_id, table_name) in enumerate(table_info):
            print(f"Processing table: {table_name} with id: {table_id}")

            entry = self.get_nth_entry(table_name, n)
            column_data_types = self.get_column_data_types(table_name)

            if ignore_description:
                ignore_description = ignore_description.lower()

            try:
                result = {}
                for col, data_type in column_data_types.items():
                    if entry and not entry.get("error"):
                        value = entry.get("results", {}).get(col)
                    else:
                        value = None

                        if include_example_entry:
                            query = f"SELECT `{col}` FROM `{self.project_id}.{self.dataset_id}.{table_name}` WHERE `{col}` IS NOT NULL LIMIT 1"
                            print(f"Executing query: {query}")
                            query_job = self._client.query(query)
                            try:
                                value = next(query_job.result())[0]
                            except StopIteration:
                                value = None

                    col_data = {}
                    if include_example_entry:
                        col_data["example_entry"] = value

                    if include_data_type:
                        col_data["data_type"] = data_type

                    if include_categories and data_type in ["STRING", "BYTES"]:
                        if data_type == "ARRAY":
                            col_data["categories"] = "Skipped for ARRAY column"
                        else:
                            try:
                                # query = f"SELECT DISTINCT `{col}` FROM `{self.project_id}.{self.dataset_id}.{table_name}`"
                                # Use random subsample
                                query = f"""
                                        SELECT `{col}`
                                        FROM (
                                            SELECT DISTINCT `{col}`
                                            FROM `{self.project_id}.{self.dataset_id}.{table_name}`
                                            WHERE `{col}` IS NOT NULL
                                            ORDER BY RAND()
                                            LIMIT 10000
                                        ) AS sub
                                        """
                                print(f"Executing query: {query}")
                                query_job = self._client.query(query)
                                distinct_values = [
                                    row[col] for row in query_job.result()
                                ]
                                distinct_values = [v for v in distinct_values if v]

                                if len(distinct_values) <= max_categories:
                                    if len(distinct_values) > max_categories:
                                        distinct_values = distinct_values[
                                            :max_categories
                                        ]
                                    distinct_values = [
                                        val[:max_cat_length]
                                        if len(val) > max_cat_length
                                        else val
                                        for val in distinct_values
                                    ]
                                    col_data["categories"] = distinct_values
                                else:
                                    col_data[
                                        "categories"
                                    ] = "Skipped due to exceeding max_categories"
                            except Exception as e:
                                print(
                                    f"Error retrieving categories for column '{col}': {str(e)}"
                                )
                                col_data[
                                    "categories"
                                ] = "Error: Failed to retrieve categories"

                    if include_nonnull_count:
                        query = f"SELECT COUNT(*) FROM `{self.project_id}.{self.dataset_id}.{table_name}` WHERE `{col}` IS NOT NULL"
                        print(f"Executing query: {query}")
                        query_job = self._client.query(query)
                        result_iterator = query_job.result()
                        col_data["n_values_nonnull"] = next(result_iterator)[0]
                        try:
                            col_data["n_values_nonnull"] = next(result_iterator)[0]
                        except StopIteration:
                            col_data["n_values_nonnull"] = None

                    if include_unique_count and data_type != "ARRAY":
                        try:
                            query = f"SELECT COUNT(DISTINCT `{col}`) FROM `{self.project_id}.{self.dataset_id}.{table_name}`"
                            print(f"Executing query: {query}")
                            query_job = self._client.query(query)
                            result_iterator = query_job.result()
                            col_data["n_values_unique"] = next(result_iterator)[0]
                        except Exception as e:
                            print(f"Error calculating unique count for column '{col}': {str(e)}")
                            col_data["n_values_unique"] = None  # Using None instead of an error message

                    result[col] = col_data

                    # Update source progress based on processed columns
                    processed_columns += 1
                    progress = int(10 + (processed_columns / total_columns) * 90)
                    self.update_source_progress(progress)

            except Exception as e:
                print(f"Error getting column info: {str(e)}")
                self.update_status_dbinfo("E3")
                backend_conn.rollback()
                raise e

            backend_conn = self._connect_backend()
            backend_cursor = backend_conn.cursor()

            try:
                for col, col_data in result.items():
                    column_name = col
                    data_type = col_data.get("data_type", None)
                    column_description = col_data.get("column_description", None)
                    categories = json.dumps(col_data.get("categories", []))
                    table_info_id = table_id

                    example_entry = col_data.get("example_entry", None)
                    example_entry = str(example_entry)

                    nonnull_count = col_data.get("n_values_nonnull", None)
                    unique_count = col_data.get("n_values_unique", None)

                    check_sql = """
                    SELECT 1 FROM db_columninfo
                    WHERE column_name = %s AND table_info_id = %s
                    """

                    backend_cursor.execute(
                        check_sql,
                        (
                            column_name,
                            table_info_id,
                        ),
                    )
                    exists = backend_cursor.fetchone()

                    if not exists:
                        insert_sql = """INSERT INTO db_columninfo
                                        (column_name, data_type, column_description, categories, table_info_id,
                                         example_entry, nonnull_count, unique_count)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""

                        backend_cursor.execute(
                            insert_sql,
                            (
                                column_name,
                                data_type,
                                column_description,
                                categories,
                                table_info_id,
                                example_entry,
                                nonnull_count,
                                unique_count,
                            ),
                        )
                    else:
                        update_sql = """UPDATE db_columninfo SET 
                                        data_type = %s, column_description = %s, categories = %s, 
                                        example_entry = %s, nonnull_count = %s, unique_count = %s
                                        WHERE column_name = %s AND table_info_id = %s"""

                        backend_cursor.execute(
                            update_sql,
                            (
                                data_type,
                                column_description,
                                categories,
                                example_entry,
                                nonnull_count,
                                unique_count,
                                column_name,
                                table_info_id,
                            ),
                        )

                backend_conn.commit()

            except Exception as e:
                print(f"Error updating backend database: {str(e)}")
                self.update_status_dbinfo("E3")
                backend_conn.rollback()
                raise Exception()

            finally:
                backend_cursor.close()
                backend_conn.close()

        return result

    def run_custom_query(self, query, params=None, timeout_sec=300):
            result_dict = {"column_names": None, "results": None, "error": None}

            try:
                query = query.strip()

                if not query.lower().startswith("select"):
                    raise ValueError("Only SELECT queries are allowed")

                job_config = bigquery.QueryJobConfig(
                    query_parameters=params
                    if params
                    else [],  # Use an empty list if params is None
                    use_legacy_sql=False,
                )
                query_job = self._client.query(query, job_config=job_config)

                # Ensure the query job is complete
                try:
                    results = query_job.result(
                        timeout=timeout_sec
                    )  # Ensures the query completes
                    if query_job.done():
                        if query_job.errors:
                            result_dict["error"] = "Query completed with errors: " + str(
                                query_job.errors
                            )
                            return result_dict

                        column_names = (
                            [field.name for field in query_job.schema]
                            if query_job.schema
                            else []
                        )
                        result_rows = [dict(row) for row in results] if results else []

                        result_dict["column_names"] = column_names
                        result_dict["results"] = result_rows
                        if not result_rows:
                            result_dict["error"] = "Query completed but returned no rows"

                        # Reformat correctly
                        result_dict["column_names"] = list(result_dict["results"][0].keys())
                        result_dict["results"] = [list(d.values()) for d in result_dict["results"]]

                        # Print the reorganized result_dict
                        print(result_dict)
                    else:
                        result_dict["error"] = "Query did not complete"
                except TimeoutError as te:
                    result_dict[
                        "error"
                    ] = f"Query exceeded {timeout_sec} seconds timeout: {str(te)}"
                except Exception as ex:
                    result_dict["error"] = "Error processing query results: " + str(ex)

            except ValueError as ve:
                result_dict["error"] = str(ve)
            except Exception as e:
                result_dict["error"] = "Error executing query: " + str(e)

            return result_dict
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for interacting with Database."
    )
    parser.add_argument(
        "db_info_id", type=int, help="ID of the database info to be used"
    )
    parser.add_argument("source_id", type=int, help="ID of source associated with it")
    parser.add_argument(
        "user_id", type=int, help="ID of the user that started the integration"
    )
    
    args = parser.parse_args()
    db_info_id = args.db_info_id
    user_id = args.user_id
    source_id = args.source_id

    load_dotenv()

    backend_db_host = os.getenv("BACKEND_DB_HOST")
    backend_db_password = os.getenv("BACKEND_DB_PASSWORD")
    encryption_key = os.getenv("ENCRYPTION_KEY")

    logger.info(f"Starting db info processing for db_dbinfo.id: {db_info_id}")
    bq_integration = BigQueryIntegration(
        db_info_id=db_info_id,
        backend_db_host=backend_db_host,
        backend_db_password=backend_db_password,
        encryption_key=encryption_key,
    )
    logger.info(f"Updating the status to I (in progress)")
    bq_integration.update_status_dbinfo("I")
    logger.debug(bq_integration.get_backend_db_info())

    logger.info("Recording table names...")
    table_names = bq_integration.get_all_table_names()
    logger.debug(f"Table names are: {table_names}")
    bq_integration.record_table_names(table_names)
    
    logger.info("Recording column info...")
    bq_integration.record_table_columninfo()

    logger.info("Recording constraints...")
    bq_integration.record_constraints()

    logger.info(f"Updating the status to C (completed)")
    bq_integration.update_source_progress(100)
    bq_integration.update_status_dbinfo("C")

    # Segment track event
    logger.info(f"Sending event to Segment")
    backend_db = BackendDatabase(host=backend_db_host, password=backend_db_password)
    email = backend_db.get_username_by_user_id(user_id)
    properties = dict(
        db_type=bq_integration.db_info["db_type"],
        db_name=bq_integration.db_info["db_name"],
    )
    context = dict(traits=dict(email=email))
    track(user_id, "complete_db_integration", properties, context)
    logger.info(f"Processing all db info completed")
