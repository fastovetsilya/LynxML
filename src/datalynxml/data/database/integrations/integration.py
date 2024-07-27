import argparse
import logging
import os
import sys
from dotenv import load_dotenv

root_path = os.path.abspath(os.path.join(__file__, "../../../../../"))
sys.path.append(root_path)

from datalynxml.data.database.integrations.multi_sql import MultiDBIntegration
from datalynxml.data.database.integrations.big_query_sql import BigQueryIntegration
from datalynxml.data.database.backend_db_tools import BackendDatabase
from datalynxml.app.segment import track

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


def get_backend_db_type(db_info_id, backend_db_host, backend_db_password):
    def connect_backend(host, password):
        try:
            db = BackendDatabase(host=host, password=password)
            conn = db.connect()
        except Exception as e:
            logger.critical("Issue connecting to backend db")
            logger.error(e)
            raise Exception()
        return conn

    # Initialize the variable to store the db_type
    db_type = None

    # Connect to the backend (MySQL in this case)
    connection = connect_backend(host=backend_db_host, password=backend_db_password)
    cursor = connection.cursor()

    try:
        # Query to fetch the db_type column for the given ID
        query = "SELECT db_type FROM db_dbinfo WHERE id = %s"
        cursor.execute(query, (db_info_id,))

        # Fetch the data
        data = cursor.fetchone()

        if data:
            # Only db_type is retrieved, so it will be the first item in the tuple
            db_type = data[0]
        else:
            logger.debug(f"No db_type found for ID: {db_info_id}")
    except Exception as e:
        logger.error(f"Error fetching db_type from backend: {e}")
    finally:
        cursor.close()
        connection.close()

    return db_type


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

    # Get db type by id from the backend
    logger.info(f"Retrieving db type for db info id: {db_info_id}")
    db_type = get_backend_db_type(
        backend_db_host=backend_db_host,
        backend_db_password=backend_db_password,
        db_info_id=db_info_id,
    )
    logger.info(f"Retrieved db type for db info id {db_info_id} is: {db_type}")

    if db_type in ["postgresql", "mysql", "oracle"]:
        db_integration = MultiDBIntegration(
            db_info_id=db_info_id,
            source_id=source_id,
            backend_db_host=backend_db_host,
            backend_db_password=backend_db_password,
            encryption_key=encryption_key,
        )
    elif db_type == "bigquery":
        db_integration = BigQueryIntegration(
            db_info_id=db_info_id,
            source_id=source_id,
            backend_db_host=backend_db_host,
            backend_db_password=backend_db_password,
            encryption_key=encryption_key,
        )
    else:
        error_message = f"Database type {db_type} not supported, abort."
        logger.error(error_message)
        raise Exception(error_message)

    logger.info(f"Starting db info processing for db_dbinfo.id: {db_info_id}")

    logger.info(f"Updating the status to I (in progress)")
    db_integration.update_status_dbinfo("I")
    logger.debug(db_integration.get_backend_db_info())

    logger.info("Recording table names...")
    table_names = db_integration.get_all_table_names()
    # table_names = db_integration.get_enabled_table_names() #TODO: Re-enable when we have actual values
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
