import logging
import sqlite3
import json
import csv
import io
from pydantic import BaseModel as BaseModel_v2, Field as Field_v2

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


class TestDatabase(BaseModel_v2):
    db_path: str

    def connect(self):
        """
        Establishes a connection to the SQLite database.
        Returns:
            sqlite3.Connection: A connection object to the SQLite database.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            print(f"An error occurred while connecting to the SQLite database: {e}")
            return None

    def get_all_table_names(self):
        """
        Fetches all table names in the SQLite database.

        Returns:
            list: A list of table names.
        """
        conn = self.connect()
        if not conn:
            return "Connection to database failed"

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            # Extract table names from the query result
            table_names = [table[0] for table in tables]
        except sqlite3.Error as e:
            print(f"An error occurred while fetching table names: {e}")
            table_names = []  # Return an empty list in case of an error
        finally:
            conn.close()

        return table_names

    def get_tables_info(self):
        """
        Fetches information about all tables including table name and an empty description.

        Returns:
            str: A formatted string containing table names and empty descriptions.
        """
        table_names = self.get_all_table_names()
        if table_names == "Connection to database failed":
            return "Connection to database failed"

        tables_info = "table_name,description\r\n"
        for table in table_names:
            tables_info += f"{table},\r\n"

        return tables_info

    def get_columns_info(self):
        """
        Fetches information about all columns in all tables.

        Returns:
            str: A formatted string containing table names, column names, and empty descriptions.
        """
        table_names = self.get_all_table_names()
        if table_names == "Connection to database failed":
            return "Connection to database failed"

        columns_info = "table_name,column_name,description\r\n"
        conn = self.connect()
        if not conn:
            return "Connection to database failed"

        try:
            cursor = conn.cursor()
            for table in table_names:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                for col in columns:
                    columns_info += f"{table},{col[1]},\r\n"
        except sqlite3.Error as e:
            print(f"An error occurred while fetching column information: {e}")
        finally:
            conn.close()

        return columns_info

    def get_column_data_types(self, table_name):
        """
        Get the data types of all columns in a specified table for SQLite.

        Args:
            table_name (str): The name of the table.

        Returns:
            dict: A dictionary where keys are column names and values are data types.
        """
        conn = self.connect()
        if not conn:
            raise Exception("Connection to database failed")

        column_data_types = {}
        query = f"PRAGMA table_info({table_name})"
        try:
            cur = conn.cursor()
            cur.execute(query)
            columns_info = cur.fetchall()
            for col in columns_info:
                column_data_types[col[1]] = col[2]  # col[1] is name, col[2] is type
        except sqlite3.Error as e:
            print(f"Error fetching column data types for {table_name}: {e}")
            raise
        finally:
            conn.close()

        return column_data_types

    def get_nth_entry(self, table_name, n):
        """
        Get the nth entry from a specified table for SQLite.

        Args:
            table_name (str): The name of the table.
            n (int): The row number (1-indexed).

        Returns:
            dict: A dictionary representation of the nth row, or None if no such row exists.
        """
        conn = self.connect()
        if not conn:
            raise Exception("Connection to database failed")

        entry = None
        query = f"SELECT * FROM {table_name} LIMIT 1 OFFSET {n - 1}"
        try:
            cur = conn.cursor()
            cur.execute(query)
            row = cur.fetchone()
            if row:
                column_names = [desc[0] for desc in cur.description]
                entry = dict(zip(column_names, row))
        except sqlite3.Error as e:
            print(f"Error fetching the {n}th entry from {table_name}: {e}")
            raise
        finally:
            conn.close()

        return entry

    def get_table_columninfo(
        self,
        table_names,
        n=1,
        max_categories=100,
        max_cat_length=15,
        include_col_desc=False,
        include_categories=True,
        include_data_type=True,
        include_nonnull_count=True,
        include_unique_count=True,
        include_example_entry=True,
        ignore_description="ignore",  # Ignore columns containing this phrase in the description
    ):
        csv_output = "Column,Data Type,Example Entry,Column Description,Categories,Non-Null Count,Unique Count\n"

        for table_name in table_names:
            # print(f"Processing table: {table_name}")
            entry = self.get_nth_entry(table_name, n)
            if entry is None:
                print(f"Table {table_name} is empty.")
                continue

            column_data_types = self.get_column_data_types(table_name)
            column_data_types = {
                col: dt for col, dt in column_data_types.items() if col in entry
            }

            if ignore_description:
                ignore_description = ignore_description.lower()

            col_desc_dict = None
            if include_col_desc:
                try:
                    desc_table_name = "_" + table_name
                    col_desc_dict = self.get_nth_entry(desc_table_name, 1)
                except:
                    pass

            for col, data_type in column_data_types.items():
                col_data = {
                    "data_type": None,
                    "example_entry": None,
                    "column_description": None,
                    "categories": [],
                    "n_values_nonnull": 0,
                    "n_values_unique": 0,
                }
                if include_data_type:
                    col_data["data_type"] = data_type

                if include_example_entry:
                    value = entry[col] if entry and col in entry else None
                    if not value:
                        with sqlite3.connect(
                            self.db_path
                        ) as conn:  # Assuming self.db_path is the path to the SQLite database
                            cur = conn.cursor()
                            query = f'SELECT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL LIMIT 1'
                            cur.execute(query)
                            value = cur.fetchone()
                            if value:
                                value = value[0]
                    col_data["example_entry"] = value

                if include_col_desc and col_desc_dict and col in col_desc_dict:
                    col_desc = col_desc_dict[col]
                    if not ignore_description or (
                        ignore_description
                        and ignore_description not in col_desc.lower()
                    ):
                        col_data["column_description"] = col_desc

                if include_categories and (
                    data_type.lower()
                    in [
                        "character varying",
                        "text",
                    ]
                    or data_type.lower().startswith("char(")
                ):
                    with sqlite3.connect(self.db_path) as conn:
                        cur = conn.cursor()
                        query = f'SELECT DISTINCT "{col}" FROM "{table_name}"'
                        cur.execute(query)
                        distinct_values = [row[0] for row in cur.fetchall() if row[0]]
                        if len(distinct_values) <= max_categories:
                            if len(distinct_values) > max_categories:
                                distinct_values = distinct_values[:max_categories]
                            for i, val in enumerate(distinct_values):
                                if len(val) > max_cat_length:
                                    distinct_values[i] = val[:max_cat_length]
                            # print(f"Distinct values are: {distinct_values}")
                            col_data["categories"] = ", ".join(distinct_values)

                if include_nonnull_count:
                    with sqlite3.connect(self.db_path) as conn:
                        cur = conn.cursor()
                        query = f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NOT NULL'
                        cur.execute(query)
                        col_data["n_values_nonnull"] = cur.fetchone()[0]

                if include_unique_count:
                    with sqlite3.connect(self.db_path) as conn:
                        cur = conn.cursor()
                        query = f'SELECT COUNT(DISTINCT "{col}") FROM "{table_name}"'
                        cur.execute(query)
                        col_data["n_values_unique"] = cur.fetchone()[0]

                # Formatting CSV output
                from ast import literal_eval

                csv_row = [
                    col,
                    col_data.get("data_type", ""),
                    col_data.get("example_entry", ""),
                    col_data.get("column_description", "")
                    if col_data.get("column_description") is not None
                    else "",
                    json.dumps(col_data.get("categories", ""))
                    if col_data.get("categories", "")
                    else "",
                    str(col_data.get("n_values_nonnull", "")),
                    str(col_data.get("n_values_unique", "")),
                ]
                # print(f"CSV row is: {csv_row}")
                csv_output += ",".join(map(str, csv_row)) + "\n"
        return csv_output

    def collect_constraints(self):
        """
        Collects the schema of the database, focusing on primary keys and foreign keys, and returns it as a CSV string.

        Returns:
            str: The database schema in CSV format focusing on primary keys and foreign keys.
        """
        conn = self.connect()
        if conn is None:
            return "Failed to connect to database."

        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "id",
                "column_name",
                "constraint_name",
                "referenced_column_name",
                "referenced_table_name",
                "db_info_id",
                "constraint_type",
                "table_name",
                "table_schema",
            ]
        )  # Duplicate 'table_name' as per given format

        constraint_id = 1  # Starting ID for constraints
        for table in tables:
            table_name = table[0]

            # Fetching primary key details
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            for column in columns:
                if column[5]:  # Column[5] is the "pk" (primary key) flag in the result
                    writer.writerow(
                        [
                            constraint_id,
                            column[1],
                            f"{table_name}_pk",
                            "",
                            "",
                            "",
                            "PRIMARY KEY",
                            table_name,
                            "",
                        ]
                    )
                    constraint_id += 1

            # Fetching foreign key details
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            fks = cursor.fetchall()
            for fk in fks:
                writer.writerow(
                    [
                        constraint_id,
                        fk[3],
                        f"{table_name}_ibfk_{fk[0]}",
                        fk[4],
                        fk[2],
                        "",
                        "FOREIGN KEY",
                        table_name,
                        "",
                    ]
                )
                constraint_id += 1

        csv_output = output.getvalue()
        output.close()
        return csv_output

    def run_custom_query(self, query: str, params=None, timeout_sec=30):
        result_dict = {"column_names": None, "results": None, "error": None}

        try:
            query = query.strip()

            # Safety check for SELECT queries
            if not query.lower().startswith("select"):
                raise ValueError("Only SELECT queries are allowed")

            # Remove the trailing semicolon
            if query.endswith(";"):
                query = query.rstrip(";")

            conn = self.connect()
            # Set timeout in milliseconds
            conn.execute(f"PRAGMA busy_timeout = {timeout_sec * 1000}")
            cur = conn.cursor()

            # Execute the query
            cur.execute(query, params if params else ())

            # Fetch column names
            column_names = [desc[0] for desc in cur.description]

            # Fetch results
            results = cur.fetchall()

            result_dict["column_names"] = column_names
            result_dict["results"] = results

        except sqlite3.Error as e:
            result_dict["error"] = f"An error occurred: {e}"

        finally:
            if conn:
                conn.close()

        return result_dict


if __name__ == "__main__":
    test_database = TestDatabase(db_path="/home/ilya/Desktop/california_schools.sqlite")
    table_names = test_database.get_all_table_names()
    print(f"Table names are: \n{table_names}")
    col_data_types = test_database.get_column_data_types(table_name=table_names[0])
    print(
        f"\nColumn data types for table name `{table_names[0]}` is: \n{col_data_types}"
    )
    first_entry = test_database.get_nth_entry(table_name=table_names[0], n=1)
    print(f"\nFirst entry for table name `{table_names[0]}` is: \n{first_entry}")
    table_columninfo = test_database.get_table_columninfo(table_names=table_names[:1])
    print(f"\nTable(s) `{table_names[:1]}` columninfo is: \n{table_columninfo}")
    table_constraints = test_database.collect_constraints()
    print(f"\nDatabase constraints are: \n{table_constraints}")
    all_tables_descriptions = test_database.get_tables_info()
    print(f"\nInformation about available tables is: \n{all_tables_descriptions}")
    all_columns_description = test_database.get_columns_info()
    print(f"\nInformation about available columns is: \n{all_columns_description}")
