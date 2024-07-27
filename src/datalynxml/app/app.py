import asyncio
import boto3
import csv
import json
import logging
import os
import sys
from threading import Thread
import time
import uuid
import websockets
from io import StringIO
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.websockets import WebSocket
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.endpoints import WebSocketEndpoint

# root_path = os.path.abspath(os.path.join(__file__, "../../../.."))
# sys.path.append(root_path)

from datalynxml.data.database.backend_db_tools import BackendDatabase
from datalynxml.data.database.integrations.multi_sql import MultiDBIntegration
from datalynxml.genieml.chat import Chat as ChatInstance


logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


async def chat(request: Request):
    try:
        data = await request.json()
        chat_id = int(data.get("chat_id", 0))
        parent_keys = data.get("parent_keys", "").split(", ")
        user_message = data.get("user_message", "")
        chat_instance = ChatInstance(chat_id=chat_id, parent_keys=parent_keys)
        response = chat_instance.get_next_response(user_message)
        return JSONResponse({"answer": response}, status_code=200)
    except Exception as e:
        return JSONResponse({}, status_code=500)


async def check_db_health(request):
    test_param = request.query_params.get("test")

    if test_param == "db":
        conn = BackendDatabase()
        db_health_status = conn.is_healthy()

        if db_health_status:
            response = {"response": "Database is healthy"}
            status = 200
        else:
            response = {"response": "Database is not healthy"}
            status = 500
    elif test_param == "logger":
        logger.debug("logger.debug")
        logger.info("logger.info")
        logger.warning("logger.warning")
        logger.error("logger.error")

        response = {"response": "Triggered all log types"}
        status = 200
    else:
        response = {"response": "Server is up"}
        status = 200

    return JSONResponse(response, status_code=status)


async def get_progress(request):
    process_id = request.query_params.get("process_id")
    page_writer = PageWriter(DB_CREDENTIALS, doc_base_directory="./tmp/")
    progress = page_writer.retrieve_process_completion(process_id)

    return JSONResponse({"progress": progress}, status_code=200)


async def add_source(request):
    try:
        data = await request.json()
        url = data.get("parent_key", "")
        page_writer = PageWriter(DB_CREDENTIALS, doc_base_directory="./tmp/")
        process_id = int(time.time() * 1000000)
        child_urls = page_writer.get_child_urls(
            url_parent=url, process_id=process_id, depth=2
        )
        page_writer.process_save_page_children_split(
            url_parent=url,
            child_urls=child_urls,
            process_id=process_id,
            email_creator="ilya@docugenie.io",
        )
        return JSONResponse({"response": process_id}, status_code=200)
    except Exception as e:
        return JSONResponse({}, status_code=500)


async def test_db_connection(request: Request):
    try:
        data = await request.json()
        db_type = data["db_type"]
        host = data["host"]
        port = data["port"]
        user = data["user"]
        password = data["password"]
        database = data["database"]
        schema = data["schema"]

        # Create an instance of MultiDBIntegration and call test_connections method
        db_integration = MultiDBIntegration(db_info_id=None)
        success, error = db_integration.test_connections(
            db_type, host, port, user, password, database, schema
        )

        if success:
            return JSONResponse({"status": "success"}, status_code=200)
        else:
            return JSONResponse({"status": "failure", "error": error}, status_code=400)
    except Exception as e:
        return JSONResponse({"status": "failure", "error": str(e)}, status_code=500)


def execute_sql_query(sql_query, db_info_id):
    db_integration = MultiDBIntegration(db_info_id=db_info_id)
    query_output = db_integration.run_custom_query(query=sql_query)
    return query_output


# Function to upload file to S3 and set expiration
def upload_to_s3_and_set_expiration(
    file_content, bucket_name, file_name, expiration_days
):
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket_name, Key=file_name, Body=file_content)
    # Set expiration
    s3.put_object_tagging(
        Bucket=bucket_name,
        Key=file_name,
        Tagging={"TagSet": [{"Key": "expire-after", "Value": str(expiration_days)}]},
    )
    return f"https://{bucket_name}.s3.amazonaws.com/{file_name}"


async def download_csv_endpoint(request: Request):
    try:
        data = await request.json()
        db_info_id = data.get("db_info_id", "")
        sql_query = data.get("sql_query", "")
        # Execute SQL query and get the output
        query_output = execute_sql_query(sql_query=sql_query, db_info_id=db_info_id)

        # Check for errors in query_output
        if query_output.get("error"):
            return JSONResponse({"error": "SQL query failed"}, status_code=500)

        # Extract column names and results
        column_names = query_output["column_names"]
        results = query_output["results"]

        # Create a CSV from the query output
        output = StringIO()
        writer = csv.writer(output)

        # Write the header
        writer.writerow(column_names)

        # Write the data rows
        writer.writerows(results)

        # Get the CSV content
        csv_content = output.getvalue()
        output.close()

        # Generate a unique filename
        file_name = f"{uuid.uuid4()}.csv"

        # Upload to S3 and get the URL
        s3_url = upload_to_s3_and_set_expiration(
            csv_content, "data-tables-csv-staging", file_name, expiration_days=7
        )  # Example expiration of 7 days

        # Return the S3 URL
        return JSONResponse({"s3_url": s3_url}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


routes = [
    Route("/health-check", check_db_health, methods=["GET"]),
    Route("/get_progress", get_progress, methods=["GET"]),
    Route("/test-db-connection", test_db_connection, methods=["POST"]),
    Route("/download_csv", download_csv_endpoint, methods=["GET"]),
]


def startup():
    logger.debug("Ready to go")


app = Starlette(debug=True, routes=routes, on_startup=[startup])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your allowed origins
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.websocket_route("/ws")
class Chat(WebSocketEndpoint):
    counter = 0
    encoding = "text"

    def __init__(self, scope, receive, send):
        super().__init__(scope, receive, send)
        self.task_event_loop = asyncio.new_event_loop()
        t = Thread(
            target=self.start_background_loop, args=(self.task_event_loop,), daemon=True
        )
        t.start()

    async def on_connect(self, websocket):
        await websocket.accept()
        logger.info(f"{websocket.client.host} connected")

    async def on_disconnect(self, websocket, close_code):
        if self.task_event_loop:
            self.task_event_loop.stop()
            self.task_event_loop = None

    async def on_receive(self, websocket: WebSocket, data):
        try:
            chat_id = websocket.query_params["chat_id"]
            user_id = websocket.query_params["user_id"]
            db_id = websocket.query_params["db_id"]
            docs_parent_keys = websocket.query_params["docs_parent_keys"]
            web_parent_keys = websocket.query_params["web_parent_keys"]

            logger.info(f"chat_id={chat_id}")
            logger.info(f"user_id={user_id}")
            logger.info(f"db_id={db_id}")
            logger.info(f"docs_parent_keys={docs_parent_keys}")
            logger.info(f"web_parent_keys={web_parent_keys}")
        except Exception as e:
            logger.exception("Error getting query params")

        try:
            asyncio.run_coroutine_threadsafe(
                self.answer_process(
                    websocket,
                    data,
                    chat_id,
                    user_id,
                    db_id,
                    docs_parent_keys,
                    web_parent_keys,
                ),
                self.task_event_loop,
            )
        except websockets.ConnectionClosed as e:
            logger.info(f"{websocket.client.host} left")
            logger.exception("Connection closed")
        except Exception as e:
            logger.exception("Other exception")

    def start_background_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def answer_process(
        self,
        websocket: WebSocket,
        message,
        chat_id,
        user_id,
        db_id,
        docs_parent_keys,
        web_parent_keys,
    ):
        try:
            # TODO: update ChatInstance initialization with the actual values
            chat_instance = ChatInstance(
                user_message=message,  # Message now goes here, no more message in get_next_response()
                chat_id=chat_id,
                user_id=user_id,  # Used only with documents, ignored for now
                db_id=db_id,
                docs_parent_keys=docs_parent_keys,  # Comma-separated parent keys for documents, or None. Ignored for now.
                web_parent_keys=web_parent_keys,  # Comma-separated parent keys for web pages, or None. Ignored for now.
            )
            logger.debug(f"Start making response for chat_id: {chat_id}")
            response = chat_instance.get_next_response()
            logger.debug(f"End making response for chat_id: {chat_id}")
            response_data = {"chat_id": chat_id, "text": response}

            # Convert the dictionary to a JSON string
            response_json = json.dumps(response_data)
        except Exception as e:
            logging.error(e, exc_info=True)
            response_data = {"chat_id": chat_id, "text": "Error getting answer"}
            await websocket.send_text(response_json)
        await websocket.send_text(response_json)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
else:
    gunicorn_logger = logging.getLogger("gunicorn.error")
    logger.handlers = gunicorn_logger.handlers
    if len(logger.handlers) > 0:
        logger.handlers[0].setFormatter(
            logging.Formatter("%(levelname)-8s [%(name)s]: %(message)s")
        )
        logger.setLevel(os.environ.get("LOG_LEVEL"))
