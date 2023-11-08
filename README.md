# Novel View Synthesis in Scenes in the Wild

Repo for project NVS in scenes in the wild.

## 1. Set up wandb (https://docs.wandb.ai/quickstart)

Sign up for a free account and login to your wandb account.

```
wandb login
```

Paste the API key from https://wandb.ai/authorize when prompted.

## 2. Download the dataset (https://kaldir.vc.in.tum.de/scannetpp/)

Register on the website and follow the download script.

## 3. Set up the environment

Clone repository and set up virtual environment.

```
git clone https://github.com/michaelnoi/scene_nvs.git
cd scene_nvs
```

```
pyenv virtualenv 3.10.10 nvs
pyenv local nvs
```

Install packages with poetry.

```
poetry install
```
