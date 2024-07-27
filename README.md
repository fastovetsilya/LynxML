# Docugenie ML toolbox#

### Current repository tree ###

```
.
├── data
│   ├── databases
│   │   ├── responses_database.db
│   │   └── signups_database.db
│   ├── logs
│   │   └── responses
│   │       └── sample_log.sample
│   └── scraping
│       ├── example.pdf
│       └── sample_scraped_data.sample
├── README.md
├── requirements.txt
└── src
    └── docugenie_ml
        ├── app
        │   ├── home_handler.py
        │   ├── __init__.py
        │   ├── settings_handler.py
        │   └── signups_handler.py
        ├── data
        │   ├── database
        │   │   ├── database_explorer.ipynb
        │   │   ├── __init__.py
        │   │   ├── pages_db_tools.py
        │   │   ├── responses_db_tools.py
        │   │   └── signups_db_tools.py
        │   └── scraping
        │       ├── google_drive
        │       │   ├── client_secret.json
        │       │   ├── download.py
        │       │   └── requirements.txt
        │       ├── google_drive_scrapers.py
        │       ├── __init__.py
        │       ├── pdf_parsers.py
        │       ├── reddit
        │       │   ├── extract_users.sh
        │       │   ├── mass_dm.py
        │       │   ├── message.txt
        │       │   ├── posts.csv
        │       │   ├── praw.ini
        │       │   ├── reddit_comment_scraper.ipynb
        │       │   ├── reddit_comments.csv
        │       │   ├── reddit_links.csv
        │       │   ├── scrape_subredits.py
        │       │   └── users.txt
        │       └── scrapers.py
        ├── __init__.py
        ├── processors
        │   ├── __init__.py
        │   ├── page_processor.py
        │   ├── questions_processor.py
        │   ├── responses_processor.py
        │   └── signups_processor.py
        └── text
            ├── __init__.py
            ├── text_completion
            │   ├── __init__.py
            │   └── openai_completion
            │       ├── finetuning
            │       │   └── gpt3
            │       │       └── gpt_finetuner.py
            │       ├── openai_completion_config.ini
            │       └── openai_completion.py
            ├── text_vectorization
            │   ├── __init__.py
            │   └── openai_vectorization
            │       ├── openai_vectorization_config.ini
            │       └── openai_vectorization.py
            └── utils.py

```

### What is this repository for? ###
This repository contains all the necessary ML tools to support Docugenie web app functionality. 
This includes data scraping, pre-processing, analysis and storage.  

### How do I get set up? ###
Get started by installing the requirements with ```pip install -r requirements.txt```. 
Place the project in the same directory where the App is located (not inside the app project). 

### Contribution guidelines ###
This repository is intended to be used together with [Docugenie app repository](https://bitbucket.org/docugenie/docugenie/src/main/).
Place two projects together in the same directory, and establish the links between them. 

Docugenie application should interact with ML toolkit through the API.
The API is located in ```src/docugenie_ml/app/``` directory. 
**All interation between the App and this ML toolkit should be done through this API!** 
The API is structured by the groups that correspond to the windows in Docugenie App.
For example, ```home_handler.py``` is responsible for handling requests from the Home page. 

If you are working on the App and the required method is not implemented in the API, please create a dummy method in the corresponding file. 
Use the dummy responses to continue working on your project, and the ML team will work on the implementation of the required methods. 
This way, the work on the App part and ML part would be efficiently separated. 
 
Example:

```python
class HomeHandler():
    def __init__(self):
        self.text_completion = OpenAITextCompletion()
        self.question = None
        self.response = None

    # TODO: implement this method later
    def process_question(self, question): 
        """ 
        NOT IMPLEMENTED.
        (Describe what should be implemented in details).
        """
        print("The method is not implemented yet.")
        dummy_answer = "The method is not implemented yet. This is a dummy answer to your question."
        return dummy_answer 
```

All the changes should be commited to a separate branch and subject to code review. 
When creating a new branch, make sure the name is easy to understand and contains Jira ticket id. 
The changes should be documented in the code with the comments and docstrings. 
Use Black (```pip install black```) to format Python script according to PEP8 formatting style before pushing the changes. 
For other languages, use commonly used standards.
Make sure to maintain the same style of writing throughout the code.

### Who do I talk to? ###

* Repo owner or admin: ilya@docugenie.io, aleks@docugenie.io
* Other community or team contact: Slack

## Run tests cases

* Install the testing library `deepeval` by doing `pip install -r requirements-dev.txt`
* Run the only test file we have (for now) by running `deepeval test run tests/test_model_t1.pyal `
* Check out the docs [here](https://docs.confident-ai.com/docs/getting-started) 
