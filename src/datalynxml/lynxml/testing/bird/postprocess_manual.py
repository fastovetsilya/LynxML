import json
import logging
import numpy as np
import pandas as pd


logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    input_path = "test_output_autochecked.csv"
    output_path = "test_output_autochecked.csv"

    test_out_df = pd.read_csv(input_path, index_col=0)

    if "res_match_manual" not in test_out_df.columns:
        test_out_df["res_match_manual"] = None

    for index, row in test_out_df.iterrows():
        gt = json.loads(row["gt"])
        pred = json.loads(row["pred"])
        res_match_auto = row["res_match_auto"]
        res_match_man = row["res_match_manual"]

        if gt["query_data"]["error"]:
            continue

        if res_match_auto == 1:
            match_man = 1
            test_out_df.loc[index, "res_match_manual"] = match_man
            test_out_df.to_csv(output_path)

        if res_match_auto == 0 and res_match_man not in [0, 1]:
            print("=" * 100)
            print(f"\nGround truth is: {gt}")
            print(f"\nPredicted is: {pred}")
            match_man = None
            while match_man not in [0, 1]:
                match_man = int(input("Is this a match (0/1)?"))
            test_out_df.loc[index, "res_match_manual"] = match_man
            test_out_df.to_csv(output_path)

        execution_accuracy = round(np.mean(test_out_df["res_match_manual"]) * 100, 4)
        logger.debug(f"Execution accuracy is {execution_accuracy} %")
