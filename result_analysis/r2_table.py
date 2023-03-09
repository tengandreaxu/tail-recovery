import os
import pandas as pd
from util.dataframes_handling import load_all_pickles_in_folder

res_dir = os.environ["RES_DIR"]
net_results = "rolling_pred"
ols_results = "rolling_ols"
lasso_results = "rolling_lasso"


def vanilla_r2(df: pd.DataFrame) -> float:
    return 1 - ((df.y - df.pred) ** 2).sum() / ((df.y - df.y.mean()) ** 2).sum()


def parse_parameters(folder_name: str) -> dict:

    output = {}
    name = None
    for parameters in folder_name.split("_"):
        if (
            parameters.startswith("lag")
            or parameters.startswith("earnings")
            or parameters == "iv"
            or parameters.startswith("type")
            or parameters == "only"
            or parameters == "final"
            or parameters == "tickers"
        ):
            continue

        if "=" in parameters:
            name, value = parameters.split("=")
            output[name] = value
        elif name != None:
            output[name] = output[name] + " " + parameters
        else:
            continue

    return output


if __name__ == "__main__":

    models = os.listdir(res_dir)
    r2s = []
    for dir in models:
        print("loading: ", dir)
        if dir.startswith("lag"):

            parameters = parse_parameters(dir)

            for results in [net_results, ols_results, lasso_results]:
                path = os.path.join(res_dir, dir, results)
                name = results.replace("rolling_", "")
                try:
                    df = load_all_pickles_in_folder(path)
                    r2 = vanilla_r2(df)
                    parameters[name] = round(r2, 3)
                    print(parameters)
                except:
                    parameters[name] = None
            r2s.append(parameters)

    df = pd.DataFrame(r2s)
    df.to_csv("r2_table.csv", index=False)
