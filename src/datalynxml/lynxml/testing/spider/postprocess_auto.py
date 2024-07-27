import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    test_out_df = pd.read_csv("test_output.csv", index_col=0)

    res_match_auto = []
    for index, row in tqdm(test_out_df.iterrows()):
        gt = json.loads(row["gt"])
        pred = json.loads(row["pred"])

        # Basic check that the exact result in present
        res_gt = gt["query_data"]["results"]
        res_pred = [p["query_data"]["results"] for p in pred if p["query_data"]]

        if res_gt in res_pred:
            res_match_auto.append(1)
        else:
            res_match_auto.append(0)

    # Print the results
    execution_accuracy = round(np.mean(res_match_auto) * 100, 4)
    logger.debug(f"Execution accuracy is {execution_accuracy} %")

    # Save the df in csv
    test_out_df["res_match_auto"] = res_match_auto
    test_out_df.to_csv("test_output_autochecked.csv")


