import os
import pandas as pd


def parse_parameters(folder_name: str) -> dict:

    output = {}
    name = None
    type_ = ""
    if "both" in folder_name:
        type_ = "both"
    elif "call" in folder_name:
        type_ = "call"
    else:
        type_ = "put"
    output["options"] = type_
    for parameters in folder_name.split("_"):
        if (
            parameters == "otm"
            or parameters == "is"
            or parameters == "t=5"
            or parameters == "call"
            or parameters == "put"
            or parameters == "both"
            or parameters == "only"
        ):
            continue

        if "=" in parameters:
            name, value = parameters.split("=")

            output[name] = eval(value)
        elif name != None:
            output[name] = output[name] + " " + parameters
        else:
            continue

    return output


if __name__ == "__main__":
    csvs_dir = "paper/res_paper/otm_study/"
    results = os.listdir(csvs_dir)
    results = [x for x in results if ".csv" in x]
    r2s_list = []
    for params in results:
        print("loading: ", params)

        parameters = parse_parameters(params.replace(".csv", ""))

        path = os.path.join(csvs_dir, params)
        r2s = pd.read_csv(path, names=["r2", "value"])
        r2s = r2s[~r2s.r2.isna()]
        r2s.value = r2s.value.apply(lambda x: round(x, 3))
        r2s.index = r2s.r2
        r2s.pop("r2")
        r2_dict = r2s.to_dict()["value"]
        parameters.update(r2_dict)

        print(parameters)

        r2s_list.append(parameters)

    df = pd.DataFrame(r2s_list)
    df.to_csv("dnn_r2_table.csv", index=False)
