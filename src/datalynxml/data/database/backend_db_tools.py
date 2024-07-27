import os
import csv
import json
import logging
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
from io import StringIO
from pydantic import BaseModel as BaseModel_v2, Field as Field_v2

logger = logging.getLogger(__name__)


class BackendDatabase(BaseModel_v2):
    """
    A class representing a MySQL database for storing web page information and process completion data.
    Attributes:
        host (str): The hostname of the MySQL server.
        user (str): The username to connect to the MySQL server.
        password (str): The password to connect to the MySQL server.
        database (str): The name of the MySQL database.
    """

    host: str
    password: str
    port: int = 5432
    user: str = "postgres"
    database: str = "postgres"

    def __init__(self, **data):
        super().__init__(**data)

    def connect(self):
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.database,
        )

    def enable_extension(self, extension_name):
        """Enable a PostgreSQL extension.

        Args:
            extension_name (str): The name of the extension to enable.
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE EXTENSION IF NOT EXISTS {extension_name}")
        print(f"Extension '{extension_name}' is enabled.")

    def is_healthy(self):
        """
        Performs a health check on the database connection.

        Returns:
            bool: True if the connection is healthy, False otherwise.
        """
        try:
            # Try to connect to the database
            with self.connect() as conn:
                # Try to execute a simple query
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                # If all the above operations were successful, return True
            return True

        except Exception as e:
            # If anything went wrong, log the exception and return False
            logger.exception(f"Database connection health check failed")
            raise e

    def run_custom_query(self, query):
        """
        Executes a custom query on the database.

        Args:
            query (str): The query to execute.

        Returns:
            result: The result of the query execution.
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(query)

                # Fetch all the rows in a list of lists and return the result.
                result = cursor.fetchall()
                return result

        except Exception as e:
            logger.exception(f"Failed to run the query")
            raise e

    def download_chat_history(self, chat_id):
        connection = self.connect()
        cursor = connection.cursor()

        query = (
            "SELECT sender, text, created_at FROM chats_message "
            "WHERE chat_id = %s ORDER BY created_at ASC"
        )
        cursor.execute(query, (chat_id,))

        chat_history = []
        for sender, text, created_at in cursor:
            message = {"role": sender, "content": text}
            chat_history.append(message)

        cursor.close()
        connection.close()

        return chat_history

    def upload_chat_message(self, chat_id, role, content, starred=False):
        connection = self.connect()
        cursor = connection.cursor()

        query = (
            "INSERT INTO chats_message (sender, text, starred, created_at, chat_id, show_in_chat) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )

        # Use the current time for 'created_at'
        created_at = datetime.now()

        cursor.execute(query, (role, content, starred, created_at, int(chat_id), False))

        # Commit the transaction
        connection.commit()

        cursor.close()
        connection.close()

    def add_update_chat_meta(self, chat_id, chat_meta):
        """
        Add new chat metadata for the specified chat id.

        If an entry with the same chat_id already exists, the chat_meta will be updated.

        CREATE TABLE chats_meta (
            chat_id INT PRIMARY KEY,
            chat_meta TEXT
        );

        Parameters:
            chat_id (str): The chat id to which the metadata will be associated.
            chat_meta (dict): The chat metadata to be stored.

        Returns:
            bool: True if the metadata was successfully added or updated, False otherwise.
        """
        try:
            chat_meta = json.dumps(chat_meta)
            with self.connect() as conn:
                cursor = conn.cursor()

                # Check if the chat_id already exists
                cursor.execute(
                    "SELECT 1 FROM chats_meta WHERE chat_id = %s", (chat_id,)
                )
                exists = cursor.fetchone()

                if exists:
                    # If the chat_id exists, update the chat_meta
                    sql_query = (
                        "UPDATE chats_meta SET chat_meta = %s WHERE chat_id = %s"
                    )
                    values = (chat_meta, chat_id)
                else:
                    # If the chat_id doesn't exist, insert a new row
                    sql_query = (
                        "INSERT INTO chats_meta (chat_id, chat_meta) VALUES (%s, %s)"
                    )
                    values = (chat_id, chat_meta)

                cursor.execute(sql_query, values)
                conn.commit()
                return True

        except Exception as e:
            logger.exception(f"Failed to add/update chat meta")
            raise e

    def get_chat_meta(self, chat_id):
        """
        Get chat metadata for the specified chat id.

        Parameters:
            chat_id (str): The chat id for which the metadata will be retrieved.

        Returns:
            str or None: The chat metadata if found, or None if no metadata exists for the chat id.
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                sql_query = "SELECT chat_meta FROM chats_meta WHERE chat_id = %s"
                values = (chat_id,)
                cursor.execute(sql_query, values)
                result = cursor.fetchone()
                result = json.loads(result[0]) if result else None
                return result

        except Exception as e:
            logger.exception(f"Failed to get chat meta")
            raise e

    def upsert_langchain_chat_memory_msgs(
        self, chat_id, chat_memory_msgs_json, chat_memory_summary
    ):
        try:
            with self.connect() as conn:
                cur = conn.cursor()

                cur.execute(
                    "UPDATE chats_langchainmemory SET chat_memory_messages = %s, chat_memory_summary_buffer = %s WHERE chat_id = %s",
                    (chat_memory_msgs_json, chat_memory_summary, chat_id),
                )

                if cur.rowcount == 0:
                    cur.execute(
                        "INSERT INTO chats_langchainmemory (chat_id, chat_memory_messages, chat_memory_summary_buffer) VALUES (%s, %s, %s)",
                        (chat_id, chat_memory_msgs_json, chat_memory_summary),
                    )

                conn.commit()

        except Exception as e:
            logger.exception("Failed to upload langchain chat memory messages")
            raise e

    def get_langchain_chat_memory(self, chat_id):
        """
        CREATE TABLE chat_table (
            chat_id SERIAL PRIMARY KEY,
            chat_memory_messages TEXT,
            chat_memory_summary_buffer TEXT
        );
        """
        try:
            with self.connect() as conn:
                cur = conn.cursor()

                # Fetch the latest chat history and summary for the user
                cur.execute(
                    "SELECT chat_memory_messages, chat_memory_summary_buffer FROM chats_langchainmemory WHERE chat_id = %s",
                    (chat_id,),
                )
                chat_memory_row = cur.fetchone()
                if chat_memory_row:
                    chat_memory_msgs_json, chat_memory_summary = chat_memory_row
                else:
                    chat_memory_msgs_json, chat_memory_summary = None, None

                return chat_memory_msgs_json, chat_memory_summary
        except Exception as e:
            logger.exception(f"Failed to download langchain chat memory messages")
            raise e
    
    def get_db_constraints(self, db_info_id):
        """
        Retrieve constraints for a given db_info_id formatted as a .csv string, including table names instead of table_info_id.

        Args:
            db_info_id (int): The ID for which constraints need to be fetched.

        Returns:
            str: The constraints data in .csv format, with table names included.
        """
        # Establish connection
        conn = self.connect()
        cursor = conn.cursor()

        # Adjusted query to join db_constraintsinfo with db_tableinfo
        # Explicitly selecting desired columns from db_constraintsinfo, excluding 'table_info_id'
        # and including 'table_name' from db_tableinfo
        query = """
        SELECT c.constraint_name, c.constraint_type, t.table_name, c.column_name, c.referenced_table_name, c.referenced_column_name
        FROM db_constraintsinfo c
        JOIN db_tableinfo t ON c.table_info_id = t.id
        WHERE c.db_info_id = %s
        """

        cursor.execute(query, (db_info_id,))

        # Fetch column names
        column_names = [desc[0] for desc in cursor.description]

        # Fetch the rows
        rows = cursor.fetchall()

        # Convert the data into a CSV string
        output = StringIO()
        csv.writer(output).writerow(column_names)  # Write column names
        csv.writer(output).writerows(rows)

        # Close the cursor and connection
        cursor.close()
        conn.close()

        return output.getvalue()

    def get_tables_with_descriptions(self, db_info_id, table_names=None):
        """
        Retrieve table names along with their descriptions for a given db_info_id formatted as a .csv string.

        Args:
            db_info_id (int): The ID for which table names and descriptions need to be fetched.
            table_names (list, optional): A list of table names to filter the results.
                                          If not provided, descriptions for all tables will be fetched.

        Returns:
            str: The table names with descriptions in .csv format.
        """
        # Establish connection
        conn = self.connect()
        cursor = conn.cursor()

        # Build the query based on the table_names parameter
        if table_names:
            table_names_placeholder = ", ".join(["%s"] * len(table_names))
            query = f"""SELECT table_name, description 
                        FROM db_tableinfo 
                        WHERE db_info_id = %s AND status = 'Y' AND table_name IN ({table_names_placeholder})"""
            cursor.execute(query, [db_info_id] + table_names)
        else:
            query = "SELECT table_name, description FROM db_tableinfo WHERE db_info_id = %s AND status = 'Y'"
            cursor.execute(query, (db_info_id,))

        # Fetch column names
        column_names = [desc[0] for desc in cursor.description]

        # Fetch the rows
        rows = cursor.fetchall()

        # Convert the data into a CSV string
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(column_names)
        writer.writerows(rows)

        # Close the cursor and connection
        cursor.close()
        conn.close()

        return output.getvalue()

    def get_column_info_for_tables(self, db_info_id, table_names):
        """
        Retrieve column info for the specified list of table names and db_info_id.

        Args:
            db_info_id (int): The ID for which column info needs to be fetched.
            table_names (list[str]): List of table names for which column info is to be retrieved.

        Returns:
            str: The column info in .csv format.
        """
        # Establish connection
        conn = self.connect()
        cursor = conn.cursor()

        # Fetch table_info_id values for the specified db_info_id and table names
        table_name_placeholders = ", ".join(["%s"] * len(table_names))
        query = f"SELECT id FROM db_tableinfo WHERE db_info_id = %s AND table_name IN ({table_name_placeholders}) AND status = 'Y'"
        cursor.execute(query, [db_info_id] + table_names)
        table_info_ids = [row[0] for row in cursor.fetchall()]

        # If there are no corresponding table_info_ids, return an empty CSV string
        if not table_info_ids:
            return ""

        # Fetch column info for the retrieved table_info_id values and join with db_tableinfo to get table_name
        table_info_id_placeholders = ", ".join(["%s"] * len(table_info_ids))
        query = f"""SELECT db_tableinfo.table_name, db_columninfo.*
                    FROM db_columninfo 
                    JOIN db_tableinfo ON db_columninfo.table_info_id = db_tableinfo.id 
                    WHERE db_columninfo.table_info_id IN ({table_info_id_placeholders})"""
        cursor.execute(query, table_info_ids)

        # Fetch column names and rows
        column_names = [desc[0] for desc in cursor.description]

        # Get the indices for 'table_info_id' and 'id' to exclude from rows
        table_info_id_index = column_names.index("table_info_id")
        id_index = column_names.index("id")

        # Remove 'table_info_id' and 'id' from column names
        column_names.remove("table_info_id")
        column_names.remove("id")

        # Adjust the rows to exclude the 'table_info_id' and 'id'
        rows = [
            tuple(
                value
                for i, value in enumerate(row)
                if i not in (table_info_id_index, id_index)
            )
            for row in cursor.fetchall()
        ]

        # Convert the data into a CSV string
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(column_names)
        writer.writerows(rows)

        # Close the cursor and connection
        cursor.close()
        conn.close()

        return output.getvalue()

    def get_username_by_user_id(self, user_id):
        """
        Get the username of a user based on their user ID.

        Parameters:
            user_id (int): The user ID for which the username will be retrieved.

        Returns:
            str or None: The username if found, or None if no user exists with the given user ID.
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                sql_query = "SELECT username FROM auth_user WHERE id = %s"
                values = (user_id,)
                cursor.execute(sql_query, values)
                result = cursor.fetchone()
                return result[0] if result else None

        except Exception as e:
            logger.exception(f"Failed to get username by user ID")
            raise e

    def create_dbexamples_table(self):
        """
        Create db_examples table with adaptive examples for generating queries
        """
        self._drop_table_if_exists("db_examples")

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE db_examples (
                        id SERIAL PRIMARY KEY,
                        db_info_id BIGINT,
                        term TEXT,
                        explanation TEXT,
                        query TEXT,
                        term_vector VECTOR(1536), 
                        FOREIGN KEY (db_info_id) REFERENCES db_dbinfo(id)
                    );
                    CREATE INDEX ON db_examples USING ivfflat (term_vector vector_cosine_ops)
                    WITH (lists = 100);
                    """
                )

        print("Table 'db_examples' created successfully!")

    def add_dbexample_entry(
        self,
        db_info_id,
        term,
        explanation,
        query,
        term_vector,
    ):
        """
        Add db_examples entry to db_examples table and return the ID of the inserted row
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO db_examples (db_info_id, term, explanation, query, term_vector)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        db_info_id,
                        term,
                        explanation,
                        query,
                        term_vector,
                    ),
                )
                inserted_id = cur.fetchone()[0]  # Fetch the returned id
        print(f"db_examples entry added successfully with ID {inserted_id}")
        return inserted_id

    def delete_dbexample_entry(self, entry_id):
        """
        Delete db_examples entry from db_examples table by ID
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM db_examples
                    WHERE id = %s
                    """,
                    (entry_id,),
                )
                # Check if the delete operation was successful
                if cur.rowcount:
                    print(f"db_examples entry with ID {entry_id} deleted successfully")
                else:
                    print(f"No db_examples entry found with ID {entry_id}.")

    def get_dbexample_by_entry_id(self, entry_id):
        """
        Retrieve a dbexample from db_examples table, as a dictionary
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM db_examples
                    WHERE id = %s
                    """,
                    (entry_id,),
                )
                result = cur.fetchone()
                if result is not None:
                    colnames = [desc[0] for desc in cur.description]

        if result:
            return dict(zip(colnames, result))
        else:
            print("No entry found.")
            return None

    def find_dbexamples_most_similar_entries(self, input_vector, db_info_id, n=1):
        """
        Find the most similar entries in the db_examples table given an input vector,
        limited to entries with the specified db_info_id.
        """
        # Convert the input_vector list to a vector in string format
        input_vector_str = str(input_vector).replace(" ", "")

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM db_examples
                    WHERE db_info_id = %s
                    ORDER BY term_vector <-> %s::vector LIMIT %s
                    """,
                    (db_info_id, input_vector_str, n),
                )
                results = cur.fetchall()

        if results:
            return [result[0] for result in results]
        else:
            print("No similar entries found.")
            return []

    def create_dbrules_table(self):
        """
        Create db_rules table with rules for generating queries
        """
        self._drop_table_if_exists("db_rules")

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE db_rules (
                        id SERIAL PRIMARY KEY,
                        db_info_id BIGINT,
                        rule TEXT,
                        FOREIGN KEY (db_info_id) REFERENCES db_dbinfo(id)
                    );
                    """
                )
        print("Table 'db_rules' created successfully")

    def add_dbrule_entry(
        self,
        db_info_id,
        rule,
    ):
        """
        Add db_rule entry to db_rules table and return the ID of the inserted row
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO db_rules (db_info_id, rule)
                    VALUES (%s, %s)
                    RETURNING id
                    """,
                    (
                        db_info_id,
                        rule,
                    ),
                )
                inserted_id = cur.fetchone()[0]  # Fetch the returned id
        print(f"db_rule entry added successfully with ID {inserted_id}")
        return inserted_id

    def delete_dbrule_entry(self, entry_id):
        """
        Delete db_rule entry from db_rules table by ID
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM db_rules
                    WHERE id = %s
                    """,
                    (entry_id,),
                )
                # Check if the delete operation was successful
                if cur.rowcount:
                    print(f"db_rule entry with ID {entry_id} deleted successfully")
                else:
                    print(f"No db_rule entry found with ID {entry_id}.")

    def get_dbrules_by_db_info_id(self, db_info_id):
        """
        Retrieve all dbrules associated with a given db_info_id from the db_rules table, as a list of dictionaries
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM db_rules
                    WHERE db_info_id = %s
                    """,
                    (db_info_id,),
                )
                results = cur.fetchall()
                if results:
                    colnames = [desc[0] for desc in cur.description]
                    return [dict(zip(colnames, result)) for result in results]
                else:
                    print("No entries found.")
                    return []

    def add_token_usage_entry(
        self,
        db_info_id,
        chat_id,
        completion_tokens,
        prompt_tokens,
        total_tokens,
        cost_completion_tokens,
        cost_prompt_tokens,
        cost_total_tokens,
    ):
        """
        Adds a new entry to the db_tokenusage table.

        Parameters:
            db_info_id (int): The ID linking to the db_dbinfo table.
            chat_id, (int): The ID of the chat.
            completion_tokens (int): The number of tokens used for completion.
            prompt_tokens (int): The number of tokens used for the prompt.
            total_tokens (int): The total number of tokens used.
            cost_completion_tokens (float): The cost associated with completion tokens.
            cost_prompt_tokens (float): The cost associated with prompt tokens.
            cost_total_tokens (float): The cost associated with total tokens.
        """
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            query = """
                INSERT INTO db_tokenusage 
                (created_at, db_info_id, chat_id, completion_tokens, prompt_tokens, total_tokens, cost_completion_tokens, cost_prompt_tokens, cost_total_tokens) 
                VALUES 
                (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            
            created_at = datetime.now()
            cur.execute(
                query,
                (
                    created_at,
                    db_info_id,
                    chat_id,
                    completion_tokens,
                    prompt_tokens,
                    total_tokens,
                    cost_completion_tokens,
                    cost_prompt_tokens,
                    cost_total_tokens,
                ),
            )
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error: {error}")
        finally:
            if conn is not None:
                conn.close()

    def clear_chat(self, chat_id):
        """
        Delete all related data in chat_message, chats_meta, and langchain_chat_memory tables based on the given chat ID.

        Parameters:
            chat_id (int): The chat ID for which data will be deleted.

        Returns:
            bool: True if deletion is successful, False otherwise.
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()

                # Delete from chat_message
                message_query = "DELETE FROM chats_message WHERE chat_id = %s"
                message_values = (chat_id,)
                cursor.execute(message_query, message_values)

                # Delete from chats_meta
                meta_query = "DELETE FROM chats_meta WHERE chat_id = %s"
                meta_values = (chat_id,)
                cursor.execute(meta_query, meta_values)

                # Delete from langchain_chat_memory
                memory_query = "DELETE FROM chats_langchainmemory WHERE chat_id = %s"
                memory_values = (chat_id,)
                cursor.execute(memory_query, memory_values)

                conn.commit()
                return True

        except Exception as e:
            logger.exception(f"Failed to delete chat data by chat ID")
            raise e


if __name__ == "__main__":
    load_dotenv()
    host = os.getenv("BACKEND_DB_HOST")
    password = os.getenv("BACKEND_DB_PASSWORD")

    # create instance of MySQLDatabase class
    backend_db = BackendDatabase(host=host, password=password)

    # Check if the connection is healthy
    if backend_db.is_healthy():
        print("The database connection is healthy.")
    else:
        print("The database connection is NOT healthy.")
