"""
This file is used to install the required packages and create the directory structure 
for the project.
"""
import os
import json

# update pip
os.system("python -m pip install --upgrade pip")

# install required packages
os.system("pip install -r requirements.txt")

# install biomed_qa as a package
os.system("pip install -e .")

# get current directory
current_dir = os.getcwd()
print(current_dir)

# get home directory
home_dir = os.path.expanduser("~")
print(home_dir)

# define directory dict
dir_dict = {
    "base_dir": f"{current_dir}",
    "data_dir": f"{current_dir}/data",
    "model_dir": f"{current_dir}/models",
    "log_dir": f"{current_dir}/logs",
    "results_dir": f"{current_dir}/results",
}

print(dir_dict)
# write dir_dict to json file
with open(f"{home_dir}/.biomedqa_dir.json", "w", encoding="utf-8") as fp:
    json.dump(dir_dict, fp)
