import logging
import os
import sys
import copy
import json
import time
from dotenv import load_dotenv
from typing import Dict, Tuple, List, Any, Optional, Union, Type
from pydantic.v1 import BaseModel, Field, constr, validator
from pydantic import (
    BaseModel as BaseModel_v2,
    Field as Field_v2,
    PrivateAttr as PrivateAttr_v2,
)

from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.schema.agent import AgentFinish
from langchain.tools import StructuredTool

root_path = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.append(root_path)

from datalynxml.lynxml.utils import intelligent_truncate_data
from datalynxml.lynxml.prompts import PLOTS_INIT_PROMPT, PLOTS_USER_MESSAGE_PROMPT

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)
load_dotenv()

# COST_PER_TOKEN_PROMPT = 0.005 / 1000  # 0.01, 0.015
# COST_PER_TOKEN_SAMPLED = 0.015 / 1000  # 0.03, 0.075

RED_START = "\033[31m"
RED_END = "\033[0m"


class LineChartDatasetInput(BaseModel):
    chart_type: constr(regex="^line$") = Field(
        ..., description="Type of the dataset, must always be 'line'"
    )
    column_name: str = Field(
        ...,
        description="Column name from the dataset corresponding to the list of data points",
    )


class BarChartDatasetInput(BaseModel):
    chart_type: constr(regex="^bars$") = Field(
        ..., description="Type of the dataset, must always be 'bars'"
    )
    column_name: str = Field(
        ...,
        description="Column name from the dataset corresponding to the list of data points",
    )


class PieChartDatasetInput(BaseModel):
    type: constr(regex="^pie$") = Field(
        ..., description="Type of the dataset, must be 'pie'"
    )
    label: str = Field(..., description="Label for the pie chart")
    data_col: List[str] = Field(
        ...,
        description="Column name(s) from the dataset corresponding to the list of data points",
    )
    backgroundColor: List[str] = Field(
        ..., description="List of hex color codes for each segment of the pie chart"
    )


class PlotDataInput(BaseModel):
    x_column: str = Field(
        ...,
        description="Column name from the dataset corresponding to the list of values for X axis",
    )
    y_columns: List[
        Union[LineChartDatasetInput, BarChartDatasetInput]
    ] = Field(..., description="List of datasets for plotting")

    @validator("y_columns", each_item=True)
    def check_data_length(cls, v, values):
        if "labels" in values and len(v.data) != len(values["labels"]):
            raise ValueError(
                "Length of data in datasets must match the length of labels"
            )
        return v


class PlotGenerator(BaseModel_v2):
    query: str
    query_data: str
    openai_api_key: str
    logger_adapter: Union[Type[logging.LoggerAdapter], None]

    # Private attributes, not part of the model schema
    _plots_init_prompt: Any = PrivateAttr_v2()  # This should be str type in the future
    _query_data_truncated: str = PrivateAttr_v2()
    _log: Type[logging.LoggerAdapter] = PrivateAttr_v2()

    class Config:
        arbitrary_types_allowed = False

    def __init__(self, **data):
        super().__init__(**data)
        self._log = (
            self.logger_adapter(logger, {"chat_id": self.chat_id})
            if self.logger_adapter
            else logger
        )
        self._log.debug("Begin PlotGenerator initialization")
        init_start_time = time.time()

        # Truncate query data
        self._query_data_truncated = json.loads(copy.deepcopy(self.query_data))
        self._query_data_truncated["results"] = intelligent_truncate_data(
            self._query_data_truncated["results"], truncate_max_res=5
        )

        # Load init prompt
        self._plots_init_prompt = copy.deepcopy(PLOTS_INIT_PROMPT)
        self._plots_user_message_prompt = copy.deepcopy(PLOTS_USER_MESSAGE_PROMPT)

        self._log.debug("Completed PlotGenerator instance init")
        self._log.debug(
            f"{RED_START}PlotGenerator initialization took the total of {time.time() - init_start_time} seconds{RED_END}"
        )

    @staticmethod
    def is_jsonified_dict(input_str):
        try:
            # Attempt to parse the string as JSON
            parsed = json.loads(input_str)
            # Check if the parsed result is a dictionary
            if isinstance(parsed, dict):
                return True
            else:
                return False
        except json.JSONDecodeError:
            # The input is not valid JSON
            return False

    @staticmethod
    def plot_data(
        x_column,
        y_columns
    ):
        chartjs_plot_command = {
            "x_column": x_column,
            "y_columns": y_columns,
        }
        return chartjs_plot_command

    def get_plot_response(self) -> str:
        self._log.debug("Start get_plot_response call")
        method_start_time = time.time()

        # Initialize the LLM
        llm = ChatOpenAI(
            temperature=0.1,
            model_name="gpt-4o",
            openai_api_key=self.openai_api_key,
        )
        tools = [
            StructuredTool.from_function(
                func=self.plot_data,
                args_schema=PlotDataInput,
                description="Function to plot data.",
            ),
        ]

        llm_with_tools = llm.bind(
            functions=[format_tool_to_openai_function(t) for t in tools]
        )

        # Using LCEL to create the agent
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{self._plots_init_prompt}",
                ),
                (
                    "user",
                    self._plots_user_message_prompt.format(
                        "{query}", "{query_data_truncated}"
                    ),
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )

        def parse(output):
            # Check if a function was invoked in the output
            if "function_call" in output.additional_kwargs:
                # Parse out the function call
                function_call = output.additional_kwargs["function_call"]
                name = function_call["name"]
                inputs = json.loads(function_call["arguments"])

                # If the function was 'ask_database', execute it and return the result
                if name == "plot_data":
                    # Assuming that 'ask_database' function is defined and can be called here
                    # You might need to adjust this part based on how your 'ask_database' function is implemented
                    result = self.plot_data(**inputs)
                    return AgentFinish(
                        return_values={"output": result}, log=str(result)
                    )
                else:
                    # Handle other functions if necessary
                    pass
            else:
                # If no function was invoked, return the original output to the user
                return AgentFinish(
                    return_values={"output": output.content}, log=output.content
                )

        agent = (
            {
                "query": lambda x: x["query"],
                "query_data_truncated": lambda x: x["query_data_truncated"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | parse
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
        )

        response = agent_executor.invoke(
            {
                "query": self.query,
                "query_data_truncated": self._query_data_truncated,
            }
        )
        response = response.get("output")

        self._log.debug(f"Return response dictionary is: {response}")
        self._log.debug(
            f"{RED_START}get_next_response took {time.time() - method_start_time} seconds{RED_END}"
        )

        return response


if __name__ == "__main__":
    example_query = "SELECT MONTH(orderDate) as Month, COUNT(DISTINCT customerNumber) as New_Customers\nFROM orders \nGROUP BY MONTH(orderDate);"
    example_query_data = json.dumps(
        {
            "column_names": ["Month", "New_Customers"],
            "results": [
                (1, 23),
                (2, 20),
                (3, 23),
                (4, 26),
                (5, 26),
                (6, 16),
                (7, 16),
                (8, 16),
                (9, 20),
                (10, 27),
                (11, 54),
            ],
            "error": None,
        }
    )
    # example_query = "Unknown"
    # example_query_data = "Unknown"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    plot_generator = PlotGenerator(
        query=example_query,
        query_data=example_query_data,
        openai_api_key=openai_api_key,
        logger_adapter=None,
    )
    response = plot_generator.get_plot_response()
