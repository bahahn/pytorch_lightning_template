# %%
import yaml


# %%
def read_yml_to_dict(path: str):
    with open(path, "r") as file:
        data = yaml.safe_load(file)
        return data


# %%
# read_yml_to_dict("source/python/configs/params.yaml")

# %%
