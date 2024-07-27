import os
import sys
import json
import glob
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

np.random.seed(1)  # Set numpy seed

root_path = os.path.abspath(os.path.join(__file__, "../../../../.."))
sys.path.append(root_path)

from datalynxml.lynxml.chat import Chat
from datalynxml.data.database.testing.bird_db_tools import TestDatabase

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    load_dotenv()

    chat_id = "143"  # replace with actual chat id
    #openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    # Load json file
    spider_dev_json_path = "/home/ilya/Desktop/SPIDER/spider/spider/dev.json"
    with open(spider_dev_json_path, "rt") as f:
        spider_dev_list = json.loads(f.read())

    # Print examples
    logger.debug(f"Loaded {len(spider_dev_list)} examples from {spider_dev_json_path}")
    logger.debug(f"First example is: {spider_dev_list[0]}")

    # Select random examples to test
    test_examples_ids = np.random.choice(len(spider_dev_list), size=200, replace=False)

    # Create list with results
    gt_queries_res = []
    generated_queries_res = []

    for test_example_id in tqdm(test_examples_ids):
        # Test example
        test_example = spider_dev_list[test_example_id]
        logger.debug(f"Analyzing example: {test_example}")
        test_question = test_example["question"]
        test_db_name = test_example["db_id"]

        # Get databases
        test_dbs_paths = os.path.join(
            "/home/ilya/Desktop/SPIDER/spider/spider/database", test_db_name
        )
        test_dbs_paths = glob.glob(os.path.join(test_dbs_paths, "*.sqlite"))

        for test_db_path in test_dbs_paths:
            # Get ground truth
            test_gt_query = test_example["query"]
            test_database = TestDatabase(db_path=test_db_path)
            try:
                test_gt_query_data = test_database.run_custom_query(query=test_gt_query)
            except:
                test_gt_query_data = {
                    "column_names": None,
                    "results": None,
                    "error": "Error: Unable to run the query",
                }
            test_gt_output = {
                "db_path": test_db_path,
                "question": test_question,
                "query": test_gt_query,
                "query_data": test_gt_query_data,
            }

            # Test configs for the chat
            test_configs = {
                "db_path": test_db_path,
                "test_question": test_question,
            }

            generated_queries = []
            try:
                chat_instance = Chat(
                    chat_id=chat_id,
                    # openai_api_key=openai_api_key,
                    anthropic_api_key=anthropic_api_key,
                    logger_adapter=None,
                    test_configs=test_configs,
                )
                for chunk in chat_instance.get_next_response():
                    if chunk:
                        chunk_dict = json.loads(chunk)
                        if "query" in chunk_dict:
                            generated_queries_output = {
                                "query": chunk_dict["query"],
                                "query_data": chunk_dict["query_data"],
                            }
                            generated_queries.append(generated_queries_output)
                        logger.debug(f"Chunk from yield: {chunk}")
                logger.debug(f"List of queries from the chat is: \n{generated_queries}")
                logger.debug(f"Ground truth query is: {test_gt_output}")

            except Exception as e:
                logger.error(f"Error running the chat: {e}")
                generated_queries_output = {
                    "query": None,
                    "query_data": None,
                    "error": f"Error: Unable to generate the query. Details: {e}",
                }
                generated_queries.append(generated_queries_output)

            # Save results as csv
            gt_queries_res.append(json.dumps(test_gt_output))
            generated_queries_res.append(json.dumps(generated_queries))
            res_df = {
                "gt": gt_queries_res,
                "pred": generated_queries_res,
            }
            res_df = pd.DataFrame.from_dict(res_df)
            res_df.to_csv("test_output.csv")
