# coarl-counterspeech

## Getting Started
1. Make sure you have `git`, `python(>=3.8, <3.10)`, [`poetry`](https://python-poetry.org/docs/#installation) installed. Preferably within a virtual environment.

2. Install dependencies
```shell
cd oarl-counterspeech
poetry install
git init
git add .
git commit -m "add: initial commit."
```

## Directory Structure

| File                                      | Description                                                                  |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| **project**                               | Main directory containing all the code            |
| **project/data**                          | Data directory containing the train, test and annotation files |
| **project/creds**                         | Directory containing all API access credentials ( project-debator / open-ai / aws)|
| **project/runs**                              | Directory to keep track of all model runs (train / eval). For each run, we store the best_model, classfication args, eval results, metrics, etc.  |
| **project/utils**                             | Program containing utility functions              |
| **project/constants**                         | Program for accessing costant variables, shared variables or default configs   |
| **CHANGELOG.md**                          | Track changes in the code, datasets, etc.                                    |
| **LICENSE**                               | Need to update  |
| **pyproject.toml**                        | Track dependencies here. Also, this means you would be using poetry.         |
| **README.md**                             | This must ring a bell.                                                       |


## Citation
If you find this repository useful in your research, please cite the following paper:

```
@misc{hengle2024intentconditioned,
      title={Intent-conditioned and Non-toxic Counterspeech Generation using Multi-Task Instruction Tuning with RLAIF}, 
      author={Amey Hengle and Aswini Kumar and Sahajpreet Singh and Anil Bandhakavi and Md Shad Akhtar and Tanmoy Chakroborty},
      year={2024},
      eprint={2403.10088},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```