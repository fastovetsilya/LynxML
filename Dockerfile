# used to run MultiDBIntegration
FROM python:3.9.17-slim

# Install dependencies
COPY requirements-multisql.txt /
RUN pip3 install -r /requirements-multisql.txt
COPY . /app
WORKDIR /app

ENTRYPOINT ["python", "/app/src/datalynxml/data/database/integrations/integration.py"]

CMD ["default_dbinfo_id", "source_id", "user_id"]