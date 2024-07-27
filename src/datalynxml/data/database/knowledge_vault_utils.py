import os
import sys
import logging
import json
from dotenv import load_dotenv

root_path = os.path.abspath(os.path.join(__file__, "../../../.."))
print(root_path)
sys.path.append(root_path)

from datalynxml.data.database.backend_db_tools import BackendDatabase
from datalynxml.text.text_vectorization.openai_vectorization import (
    OpenAITextVectorization,
)
from datalynxml.text.utils import compute_cosine_similarity

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


def add_adaptive_example(
    openai_api_key,
    backend_db_host,
    backend_db_password,
    db_info_id,
    term,
    explanation,
    query,
):
    logger.debug(
        f"Adding adaptive example for the term {term} for db_info_id {db_info_id}"
    )

    # create instance of MySQLDatabase class
    backend_db = BackendDatabase(host=backend_db_host, password=backend_db_password)
    backend_db.enable_extension("vector")

    # Create an instance of OpenAI text vectorizer
    openai_text_vectorization = OpenAITextVectorization(openai_api_key=openai_api_key)
    term_vector = openai_text_vectorization.vectorize_text(text_input=term)

    # Add the example to the database
    inserted_id = backend_db.add_dbexample_entry(
        db_info_id=db_info_id,
        term=term,
        explanation=explanation,
        query=query,
        term_vector=term_vector,
    )
    logger.debug(f"Adaptive example added successfully with id {inserted_id}")


def find_adaptive_examples(
    openai_api_key,
    backend_db_host,
    backend_db_password,
    db_info_id,
    text_input,
    n_examples,
):
    logger.debug(
        f"Searching for n={n_examples} most similar adaptive examples for the text {text_input} for db_info_id {db_info_id}"
    )
    # create instance of MySQLDatabase class
    backend_db = BackendDatabase(host=backend_db_host, password=backend_db_password)
    backend_db.enable_extension("vector")

    # Create an instance of OpenAI text vectorizer and vectorise input text
    openai_text_vectorization = OpenAITextVectorization(openai_api_key=openai_api_key)
    input_vector = openai_text_vectorization.vectorize_text(text_input=text_input)

    # Find most similar examples
    adaptive_examples_ids = backend_db.find_dbexamples_most_similar_entries(
        input_vector=input_vector, db_info_id=db_info_id, n=n_examples
    )

    if len(adaptive_examples_ids) == 0:
        logger.warning("No adaptive examples found!")
        return [], []

    adaptive_examples = [
        backend_db.get_dbexample_by_entry_id(entry_id=i) for i in adaptive_examples_ids
    ]
    for i in adaptive_examples:
        i["term_vector"] = json.loads(i["term_vector"])

    cosine_similarities = [
        compute_cosine_similarity(e["term_vector"], input_vector)
        for e in adaptive_examples
    ]
    logger.debug(
        f""""
                 Found n={n_examples} adaptive examples: {[ax['term'] for ax in adaptive_examples]}. 
                 Similarities are: {cosine_similarities}. 
                 """
    )

    return adaptive_examples, cosine_similarities


def delete_adaptive_example(
    backend_db_host,
    backend_db_password,
    entry_id,
):
    logger.debug(f"Deleting adaptive example with entry id: {entry_id}")
    # create instance of MySQLDatabase class
    backend_db = BackendDatabase(host=backend_db_host, password=backend_db_password)

    # Delete adaptive example
    backend_db.delete_dbexample_entry(entry_id=entry_id)
    logger.info("Adaptive example entry deleted successfully")


def add_rule(
    backend_db_host,
    backend_db_password,
    db_info_id,
    rule,
):
    logger.debug(f"Adding rule `{rule}` for db_info_id {db_info_id}")
    # create instance of MySQLDatabase class
    backend_db = BackendDatabase(host=backend_db_host, password=backend_db_password)

    # Add the rule to the database
    inserted_id = backend_db.add_dbrule_entry(db_info_id=db_info_id, rule=rule)
    logger.debug(f"Added the rule successfully with id {inserted_id}")


def get_rules(
    backend_db_host,
    backend_db_password,
    db_info_id,
):
    logger.debug(f"Getting the rules for db_info_id {db_info_id}")
    # create instance of MySQLDatabase class
    backend_db = BackendDatabase(host=backend_db_host, password=backend_db_password)

    # Retrieve the rules for the specified db info id
    rules = backend_db.get_dbrules_by_db_info_id(db_info_id=db_info_id)

    return rules


def delete_rule(
    backend_db_host,
    backend_db_password,
    entry_id,
):
    logger.debug(f"Deleting rule with entry id: {entry_id}")
    # create instance of MySQLDatabase class
    backend_db = BackendDatabase(host=backend_db_host, password=backend_db_password)

    # Delete rule
    backend_db.delete_dbrule_entry(entry_id=entry_id)
    logger.info("Rule entry deleted successfully")


if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    backend_db_host = os.getenv("BACKEND_DB_HOST")
    backend_db_password = os.getenv("BACKEND_DB_PASSWORD")

    # Add example
    # add_adaptive_example_input = {
    #     "openai_api_key": openai_api_key,
    #     "backend_db_host": backend_db_host,
    #     "backend_db_password": backend_db_password,
    #     "db_info_id": 23,
    #     "term": "Customer growth",
    #     "explanation": "The number of new customers per month",
    #     "query": None,
    # }
    # add_adaptive_example(**add_adaptive_example_input)

    # # Find adaptive examples
    # find_adaptive_examples_input = {
    #     "openai_api_key": openai_api_key,
    #     "backend_db_host": backend_db_host,
    #     "backend_db_password": backend_db_password,
    #     "db_info_id": 4,
    #     "text_input": "What is our customer growth?",
    #     "n_examples": 1,
    # }
    # find_adaptive_examples(**find_adaptive_examples_input)

    # Add rule
    # add_rule_input = {
    #     "backend_db_host": backend_db_host,
    #     "backend_db_password": backend_db_password,
    #     "db_info_id": 23,
    #     "rule": "Never ask clarifying questions",
    # }
    # add_rule(**add_rule_input)

    # Get all rules
    # rules = get_rules(backend_db_host=backend_db_host,
    #                   backend_db_password=backend_db_password,
    #                   db_info_id=4
    #                   )
    # logger.debug(f"Retrieved rules are: {rules}")

    # Delete adaptive example or rule
    # delete_adaptive_example(
    #     backend_db_host=backend_db_host,
    #     backend_db_password=backend_db_password,
    #     entry_id=1,
    # )
    # delete_rule(
    #     backend_db_host=backend_db_host,
    #     backend_db_password=backend_db_password,
    #     entry_id=1,
    # )
