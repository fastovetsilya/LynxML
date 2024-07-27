import argparse
import os
import json
import mysql.connector
import oracledb
import psycopg2
import logging
import sys
import uuid
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, Tuple, List, Any, Optional
from pydantic.v1 import BaseModel, Field
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


class MultiDBIntegration(BaseModel_v2):
    """
    This class provides functionality for integrating multiple SQL databases with a MySQL Docugenie backend.
    It supports operations such as connecting to databases, testing connections, updating statuses,
    recording constraints, fetching table names, and running custom queries.

    Attributes:
        backend_db_host (str): Hostname of the backend database.
        backend_db_password (str): Password for the backend database.
        db_info_id (int): Identifier for the database information.
        source_id (int): Identifier for the source associated with the db.
        encryption_key (str): Key used for encryption purposes.
        db_info (Optional[Dict]): Dictionary containing database information. Defaults to an empty dict.
        db_type (Optional[str]): Type of the database (e.g., 'postgresql', 'mysql', 'oracle'). Defaults to an empty string.
        company_info (Optional[str]): Information about the company. Defaults to an empty string.
        db_schema (Optional[str]): Schema of the database. Defaults to 'public'.

    Methods:
        __init__(**data): Initializes the MultiDBIntegration instance.
        _connect_backend(): Establishes a connection to the backend database.
        _connect_target_db(): Connects to the target database based on db_info.
        test_connections(db_type, host, port, user, password, database, schema): Tests the connection to a database.
        update_status_dbinfo(status_code): Updates the status of the database information in the backend.
        upsert_sources_source_status(status_code): Inserts or updates the source status in the backend.
        get_backend_db_info(): Retrieves database information from the backend.
        record_constraints(): Records the constraints of the connected database into the backend.
        get_all_table_names(): Retrieves all table names from the integrated database.
        get_enabled_table_names(): Fetches names of tables with 'enabled' status from the backend.
        record_table_names(table_names): Records table names into the backend database.
        get_recorded_table_info(): Retrieves recorded table information from the backend.
        get_column_data_types(table_name): Fetches data types of columns in a specified table.
        get_nth_entry(table_name, n): Retrieves the nth entry from a specified table.
        record_table_columninfo(...): Records information about table columns into the backend.
        run_custom_query(query, params=None, timeout_sec=30): Executes a custom query on the target database.

    Backend Database Legends:
        db_info status legend:
            P = pending
            C = completed
            I = in progress
            E1 = could not connect to dbinfo db
            E2 = could not connect to out backend db
            E3 = error during process

        db_tableinfo status legend:
            D = disabled
            E = enabled
    """

    backend_db_host: str
    backend_db_password: str
    db_info_id: int
    encryption_key: str

    db_info: Optional[Dict] = Field_v2(default_factory=dict)
    db_type: Optional[str] = Field_v2(default_factory=str)
    company_info: Optional[str] = Field_v2(default_factory=str)
    db_schema: Optional[str] = Field_v2(default_factory=str)
    source_id: Optional[int] = Field_v2(default_factory=int)

    class Config:
        arbitrary_types_allowed = False

    def __init__(self, **data):
        super().__init__(**data)
        self.db_info = {}

        if self.db_info_id != 0:
            self.db_info = self.get_backend_db_info()
            self.db_type = self.db_info.get("db_type", "").lower()
            self.company_info = self.db_info.get("company_description", "")

            db_schema_value = self.db_info.get("db_schema")
            if db_schema_value is None or db_schema_value == "":
                self.db_schema = "public"
            else:
                self.db_schema = db_schema_value

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

    def _connect_target_db(self):
        timeout_seconds = 4
            
        try:
            db_type = self.db_info.get("db_type", "").lower()
            db_host = self.db_info.get("db_host")
            db_port = self.db_info.get("db_port")
            db_username = self.db_info.get("db_username")
            db_password = self.db_info.get("db_password")
            db_name = self.db_info.get("db_name")
            db_schema = self.db_info.get("db_schema")

            if db_type == "postgresql":
                conn = psycopg2.connect(
                    host=db_host,
                    port=db_port,
                    user=db_username,
                    password=db_password,
                    dbname=db_name,
                    connect_timeout=timeout_seconds
                )
                if db_schema:
                    cursor = conn.cursor()
                    query = f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{db_schema}';"
                    cursor.execute(query)
                    result = cursor.fetchone()

                    if result is None:
                        raise Exception("Can't find schema")

                    # Setting the search path to use the specified schema
                    with conn.cursor() as cursor:
                        cursor.execute(f"SET search_path TO {db_schema}")

            elif db_type == "mysql":
                conn = mysql.connector.connect(
                    host=db_host,
                    port=db_port,
                    user=db_username,
                    password=db_password,
                    database=db_name,
                    connection_timeout=timeout_seconds
                )

            elif db_type == "oracle":
                dsn_str = oracledb.makedsn(
                    host=db_host,
                    port=db_port,
                    sid=db_name,
                )
                conn = oracledb.connect(
                    user=db_username,
                    password=db_password,
                    dsn=dsn_str
                )
            else:
                error_message = f"Database type {db_type} not supported."
                logger.error(error_message)
                self.update_status_dbinfo("E1")
                raise Exception(error_message)

            return conn

        except Exception as error:
            error_message = str(error)
            logger.error("Issue connecting to host db: %s", error_message)
            self.update_status_dbinfo("E1")
            raise Exception(error_message)

    def test_connections(self, db_type, host, port, user, password, database, schema):
        # Set db_info
        self.db_info = {
            "db_type": db_type,
            "db_host": host,
            "db_port": port,
            "db_username": user,
            "db_password": password,
            "db_name": database,
            "db_schema": schema,
        }

        try:
            db_conn = self._connect_target_db()
            db_conn.close()

        except Exception as error:
            return False, str(error)

        return True, None

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
                result_dict["db_host"] = decrypt(
                    enc=result_dict["db_host"], key=self.encryption_key
                )
                result_dict["db_name"] = decrypt(
                    enc=result_dict["db_name"], key=self.encryption_key
                )
                result_dict["db_username"] = decrypt(
                    enc=result_dict["db_username"], key=self.encryption_key
                )
                result_dict["db_password"] = decrypt(
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

    def get_associated_source_id(self):
        connection = self._connect_backend()
        cursor = connection.cursor()

        try:
            query = "select * from sources_source ss where ss.identifier_id = %s;"
            cursor.execute(query, (self.db_info_id,))

            data = cursor.fetchone()
            column_names = [desc[0] for desc in cursor.description]
            result_dict = dict(zip(column_names, data))

            return result_dict["id"]
        
        except Exception as e:
            logger.exception(f"Error getting source using db_info_id {self.db_info_id}")

        finally:
            cursor.close()
            connection.close()

    def record_constraints(self):
        # Connecting to the database
        integr_conn = self._connect_target_db()
        integr_cursor = integr_conn.cursor()

        # Query to fetch constraints from. This fetches foreign key constraints.
        postgresql_query = """
        SELECT
            tc.table_schema,
            tc.table_name,
            tc.constraint_name,
            tc.constraint_type,
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM 
            information_schema.table_constraints AS tc 
            JOIN information_schema.key_column_usage AS kcu 
                ON tc.constraint_name = kcu.constraint_name 
                AND tc.table_schema = kcu.table_schema
            LEFT JOIN information_schema.constraint_column_usage AS ccu 
                ON ccu.constraint_name = tc.constraint_name 
                AND ccu.table_schema = tc.table_schema
        WHERE 
            tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
            AND tc.table_schema NOT IN ('integr_catalog', 'information_schema');
        """
        mysql_query = """
        SELECT 
            tc.CONSTRAINT_SCHEMA as table_schema,
            tc.TABLE_NAME as table_name,
            tc.CONSTRAINT_NAME as constraint_name,
            tc.CONSTRAINT_TYPE as constraint_type,
            kcu.COLUMN_NAME as column_name,
            kcu.REFERENCED_TABLE_NAME as foreign_table_name,
            kcu.REFERENCED_COLUMN_NAME as foreign_column_name
        FROM 
            information_schema.TABLE_CONSTRAINTS as tc 
        JOIN 
            information_schema.KEY_COLUMN_USAGE as kcu 
            ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME 
            AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA 
        WHERE 
            tc.CONSTRAINT_TYPE IN ('PRIMARY KEY', 'FOREIGN KEY') 
            AND tc.TABLE_SCHEMA NOT IN ('integr_catalog', 'information_schema', 'mysql', 'performance_schema', 'sys');
        """
        oracle_query = """
            SELECT
                a.owner AS table_schema,
                a.table_name,
                a.constraint_name,
                CASE a.constraint_type 
                    WHEN 'P' THEN 'PRIMARY KEY'
                    WHEN 'R' THEN 'FOREIGN KEY'
                    ELSE a.constraint_type 
                END AS constraint_type,
                c.column_name,
                COALESCE(
                    (SELECT table_name 
                     FROM all_constraints 
                     WHERE owner = a.r_owner 
                     AND constraint_name = a.r_constraint_name),
                     a.table_name) AS foreign_table_name,
                COALESCE(
                    (SELECT column_name 
                     FROM all_cons_columns 
                     WHERE owner = a.r_owner 
                     AND constraint_name = a.r_constraint_name 
                     AND rownum <= 1), 
                     c.column_name) AS foreign_column_name
            FROM 
                all_constraints a
                LEFT JOIN all_cons_columns c ON a.owner = c.owner AND a.constraint_name = c.constraint_name
            WHERE 
                a.constraint_type IN ('P', 'R')
                AND a.owner NOT IN ('SYS', 'SYSTEM')
        """

        if self.db_type == "postgresql":
            query = postgresql_query
        elif self.db_type == "mysql":
            query = mysql_query
        elif self.db_type == "oracle":
            query = oracle_query
        else:
            raise Exception(
                f"Database Integration Error: db type {self.db_type} not supported."
            )

        integr_cursor.execute(query)
        constraints = integr_cursor.fetchall()
        integr_cursor.close()
        integr_conn.close()

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

    def get_all_table_names(self):
        """
        Get the names of all tables in the integrated database.

        Returns:
            list: A list of table names.
        """

        # Check if self.schema is set, else use default
        schema = self.db_schema

        postgresql_query = f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = '{schema}'
        """
        mysql_query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = DATABASE()
        """
        oracle_query = f"""
            SELECT table_name
            FROM all_tables
            WHERE owner = USER
        """

        if self.db_type == "postgresql":
            query = postgresql_query
        elif self.db_type == "mysql":
            query = mysql_query
        elif self.db_type == "oracle":
            query = oracle_query
        else:
            raise Exception(
                f"Database Integration Error: db type {self.db_type} not supported."
            )

        try:
            with self._connect_target_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    results = cur.fetchall()
                    table_names = [table[0] for table in results]

        except Exception as e:
            logger.error(f"Error fetching table names: {str(e)}")
            self.update_status_dbinfo("E3")
            raise Exception()

        return table_names

    def get_enabled_table_names(self):
        """Get table names with status 'E' from the backend database."""
        conn = self._connect_backend()
        cursor = conn.cursor()

        # Query to get table names with status "E"
        query = """
            SELECT table_name 
            FROM db_tableinfo
            WHERE db_info_id = %s AND status = 'E'
        """
        try:
            cursor.execute(query, (self.db_info_id,))
            table_names = [row[0] for row in cursor.fetchall()]

        except Exception as e:
            logger.error("Error getting enabled table names")
            logger.error(e)
            self.update_status_dbinfo("E3")
            raise Exception()

        cursor.close()
        conn.close()

        return table_names

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
                    

                # Update progress
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

    def get_column_data_types(self, table_name):
        """
        Get the data types of all columns in a specified table.

        Args:
            table_name (str): The name of the table.

        Returns:
            dict: A dictionary where keys are column names and values are data types.
        """

        postgresql_and_mysql_query = """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = %s
        """

        oracle_query = """
            SELECT column_name, data_type 
            FROM all_tab_columns 
            WHERE table_name = :1
        """

        if self.db_type in ["postgresql", "mysql"]:
            query = postgresql_and_mysql_query
        elif self.db_type == "oracle":
            query = oracle_query
        else:
            logger.error(
                f"Database Integration Error: db type {self.db_type} not supported."
            )
            self.update_status_dbinfo("E3")
            raise Exception()

        try:
            with self._connect_target_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (table_name,))
                    column_data_types = {
                        column[0]: column[1] for column in cur.fetchall()
                    }

        except Exception as e:
            logger.error(f"Error fetching column data types for {table_name}: {str(e)}")
            self.update_status_dbinfo("E3")
            raise Exception()

        return column_data_types

    def get_nth_entry(self, table_name, n):
        """
        Get the nth entry from a specified table.

        Args:
            table_name (str): The name of the table.
            n (int): The row number.

        Returns:
            dict: A dictionary representation of the nth row, or None if no such row exists.
        """
        entry = None

        postgresql_query = f'SELECT * FROM "{table_name}" OFFSET {n - 1} LIMIT 1'
        mysql_query = "SELECT * FROM `%s` LIMIT %s, 1" % (table_name, n - 1)
        oracle_query = f"""
                SELECT * FROM (
                    SELECT t.*, ROWNUM rn 
                    FROM "{table_name}" t
                ) 
                WHERE rn = {n}
            """

        if self.db_type == "postgresql":
            query = postgresql_query
        elif self.db_type == "mysql":
            query = mysql_query
        elif self.db_type == "oracle":
            query = oracle_query
        else:
            raise Exception(
                f"Database Integration Error: db type {self.db_type} not supported."
            )

        try:
            with self._connect_target_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    row = cur.fetchone()
                    if row is not None:
                        column_names = [desc[0] for desc in cur.description]
                        entry = dict(zip(column_names, row))

        except Exception as e:
            logger.error(f"Error fetching the {n}th entry from {table_name}: {str(e)}")
            self.update_status_dbinfo("E3")
            raise Exception()

        return entry

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
        ignore_description="ignore",  # Ignore columns containing this phrase in the description
    ):
        table_info = self.get_recorded_table_info()
        
        # Calculate the total number of columns to update progress accurately
        total_columns = 0
        for _, table_name in table_info:
            column_data_types = self.get_column_data_types(table_name)
            total_columns += len(column_data_types)
        processed_columns = 0  # Counter for processed columns to update progress

        for index, (table_id, table_name) in enumerate(table_info):
            logger.debug(f"Processing table: {table_name} with id: {table_id}")

            # Get entry and column data types
            entry = self.get_nth_entry(table_name, n)
            if entry is not None:
                column_data_types = self.get_column_data_types(table_name)
                # Filter column data types
                column_data_types = {
                    col: dt for col, dt in column_data_types.items() if col in entry
                }
            else:
                logger.error(f"Table {table_name} with id: {table_id} is empty.")
                continue  # Skip this table and move to the next one

            if ignore_description:
                ignore_description = ignore_description.lower()
            
            try:
                # Iterate
                result = {}
                for col, data_type in column_data_types.items():
                    if entry:
                        value = entry[col]
                    else:
                        value = None

                    # Find the first non-null value if current value is null
                    if not value and include_example_entry:  # <-- Added condition here
                        with self._connect_target_db() as conn:
                            with conn.cursor() as cur:
                                postgresql_query = f'SELECT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL LIMIT 1'
                                mysql_query = f"SELECT `{col}` FROM `{table_name}` WHERE `{col}` IS NOT NULL LIMIT 1"
                                oracle_query = f'SELECT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL AND ROWNUM <= 1'

                                if self.db_type == "postgresql":
                                    query = postgresql_query
                                elif self.db_type == "mysql":
                                    query = mysql_query
                                elif self.db_type == "oracle":
                                    query = oracle_query
                                else:
                                    raise Exception(
                                        f"Database Integration Error: db type {self.db_type} not supported."
                                    )
                                cur.execute(query)
                                value = cur.fetchone()
                                if value:
                                    value = value[0]

                    col_data = {}
                    if include_example_entry:  # <-- Added condition here
                        col_data["example_entry"] = value

                    if include_data_type:
                        col_data["data_type"] = data_type

                    # Add categories for the data
                    # Define categorical data types for each db types
                    if self.db_type == "postgresql":
                        categorical_data_types = [
                            "character varying",
                            "text",
                            "char",
                            "enum",
                        ]
                    elif self.db_type == "mysql":
                        categorical_data_types = ["varchar", "text", "char", "enum", "set"]
                    elif self.db_type == "oracle":
                        categorical_data_types = ["varchar2", "clob", "char"]
                    else:
                        raise Exception(
                            f"Database Integration Error: db type {self.db_type} not supported."
                        )
                    # Check if column is categorical
                    if include_categories and data_type.lower() in categorical_data_types:
                        with self._connect_target_db() as conn:
                            with conn.cursor() as cur:
                                postgresql_query = (
                                    f'SELECT DISTINCT {col} FROM "{table_name}"'
                                )
                                mysql_query = f"SELECT DISTINCT `{col}` FROM `{table_name}`"
                                oracle_query = f'SELECT DISTINCT "{col}" FROM "{table_name}"'

                                if self.db_type == "postgresql":
                                    query = postgresql_query
                                elif self.db_type == "mysql":
                                    query = mysql_query
                                elif self.db_type == "oracle":
                                    query = oracle_query
                                else:
                                    raise Exception(
                                        f"Database Integration Error: db type {self.db_type} not supported."
                                    )
                                cur.execute(query)
                                distinct_values = [row[0] for row in cur.fetchall()]
                                distinct_values = [v for v in distinct_values if v]
                                if len(distinct_values) <= max_categories:
                                    if len(distinct_values) > max_categories:
                                        distinct_values = distinct_values[:max_categories]
                                    for i, val in enumerate(distinct_values):
                                        if len(val) > max_cat_length:
                                            distinct_values[i] = val[:max_cat_length]
                                    col_data["categories"] = distinct_values

                    # Fetch nonnull count
                    if include_nonnull_count:
                        with self._connect_target_db() as conn:
                            with conn.cursor() as cur:
                                postgresql_query = f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NOT NULL'
                                mysql_query = f"SELECT COUNT(*) FROM `{table_name}` WHERE `{col}` IS NOT NULL"
                                oracle_query = f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NOT NULL'

                                if self.db_type == "postgresql":
                                    query = postgresql_query
                                elif self.db_type == "mysql":
                                    query = mysql_query
                                elif self.db_type == "oracle":
                                    query = oracle_query
                                else:
                                    raise Exception(
                                        f"Database Integration Error: db type {self.db_type} not supported."
                                    )
                                cur.execute(query)
                                col_data["n_values_nonnull"] = cur.fetchone()[0]

                    # Fetch unique count
                    if include_unique_count:
                        with self._connect_target_db() as conn:
                            with conn.cursor() as cur:
                                postgresql_query = (
                                    f'SELECT COUNT(DISTINCT "{col}") FROM "{table_name}"'
                                )
                                mysql_query = (
                                    f"SELECT COUNT(DISTINCT `{col}`) FROM `{table_name}`"
                                )
                                oracle_query = (
                                    f'SELECT COUNT(DISTINCT "{col}") FROM "{table_name}"'
                                )

                                if self.db_type == "postgresql":
                                    query = postgresql_query
                                elif self.db_type == "mysql":
                                    query = mysql_query
                                elif self.db_type == "oracle":
                                    query = oracle_query
                                else:
                                    raise Exception(
                                        f"Database Integration Error: db type {self.db_type} not supported."
                                    )
                                cur.execute(query)
                                col_data["n_values_unique"] = cur.fetchone()[0]

                    result[col] = col_data

                    # Update source progress based on processed columns
                    processed_columns += 1
                    progress = int(10 + (processed_columns / total_columns) * 90)
                    self.update_source_progress(progress)

            except Exception as e:
                logger.error(f"Error getting column info: {e}")
                backend_conn.rollback()  # Rollback the whole transaction if any error occurs
                self.update_status_dbinfo("E3")
                raise e

            # Connect to the backend (MySQL in this case)
            backend_conn = self._connect_backend()
            backend_cursor = backend_conn.cursor()

            # Try to update the backend database
            try:
                # Loop through the result dictionary and update backend database
                for col, col_data in result.items():
                    # Prepare data to be inserted or checked
                    column_name = col
                    data_type = col_data.get("data_type", None)
                    column_description = col_data.get("column_description", None)
                    categories = json.dumps(
                        col_data.get("categories", [])
                    )  # Convert list to json
                    table_info_id = table_id

                    example_entry = col_data.get("example_entry", None)
                    example_entry = str(example_entry)

                    nonnull_count = col_data.get("n_values_nonnull", None)
                    unique_count = col_data.get("n_values_unique", None)

                    # Check if the row already exists in db_columninfo
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

                    # If the row doesn't exist, insert it
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
                    # If the row exists, update it
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

                # Commit changes to the database after processing all columns
                backend_conn.commit()

            except Exception as e:
                logger.error(f"Error updating backend database: {e}")
                backend_conn.rollback()  # Rollback the whole transaction if any error occurs
                self.update_status_dbinfo("E3")
                raise Exception()

            finally:
                backend_cursor.close()
                backend_conn.close()

    def run_custom_query(self, query, params=None, timeout_sec=300):
        result_dict = {"column_names": None, "results": None, "error": None}

        try:
            query = query.strip()

            # Safety check for SELECT queries
            if not query.lower().startswith("select"):
                raise ValueError("Only SELECT queries are allowed")

            with self._connect_target_db() as conn:
                with conn.cursor() as cur:
                    if self.db_type == "postgresql":
                        cur.execute(f"SET statement_timeout = {timeout_sec * 1000}")
                    elif self.db_type == "mysql":
                        cur.execute(f"SET max_execution_time = {timeout_sec * 1000}")

                    # Remove the trailing semicolon
                    if query.endswith(";"):
                        query = query.rstrip(";")

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            cur.execute, query, params if params else ()
                        )
                        try:
                            future.result(timeout=timeout_sec)

                            # Fetch column names
                            column_names = [desc[0] for desc in cur.description]

                            # Fetch results
                            results = cur.fetchall()

                            result_dict["column_names"] = column_names
                            result_dict["results"] = results

                        except TimeoutError:
                            result_dict[
                                "error"
                            ] = f"Query exceeded {timeout_sec} seconds timeout"
                            logger.error(result_dict["error"])

        except Exception as e:
            result_dict["error"] = str(e)
            logger.error(
                f"Running custom query failed: {result_dict['error']}", exc_info=True
            )  # Log the full exception

        return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for interacting with Database."
    )
    parser.add_argument(
        "db_info_id", type=int, help="ID of the database info to be used"
    )
    parser.add_argument(
        "source_id", type=int, help="ID of source associated with it"
    )
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
    db_integration = MultiDBIntegration(
        db_info_id=db_info_id,
        source_id=source_id,
        backend_db_host=backend_db_host,
        backend_db_password=backend_db_password,
        encryption_key=encryption_key,
    )
    logger.info(f"Updating the status to I (in progress)")
    db_integration.update_status_dbinfo("I")
    logger.debug(db_integration.get_backend_db_info())

    logger.info("Recording table names...")
    table_names = db_integration.get_all_table_names()
    logger.debug(f"Table names are: {table_names}")
    db_integration.record_table_names(table_names)
    logger.info("Recording column info...")
    db_integration.record_table_columninfo()
    logger.info("Recording constraints...")
    db_integration.record_constraints()
    logger.info(f"Updating the status to C (completed)")
    db_integration.update_source_progress(100)
    db_integration.update_status_dbinfo("C")

    # Segment track event
    logger.info(f"Sending event to Segment")
    backend_db = BackendDatabase(host=backend_db_host, password=backend_db_password)
    email = backend_db.get_username_by_user_id(user_id)
    properties = dict(
        db_type=db_integration.db_info["db_type"],
        db_name=db_integration.db_info["db_name"],
    )
    context = dict(traits=dict(email=email))
    track(user_id, "complete_db_integration", properties, context)
    logger.info(f"Processing all db info completed")
