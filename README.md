# Emma the slack app

![showcase](https://github.com/alanahmet/gcs-slack-app/blob/main/showcase/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202023-07-21%20163550.png)
Emma the slack app is a gift recommender ai with capability of vector search on gcs database for desired gifts.Use [Google Cloude](https://colab.research.google.com/github/GoogleCloudPlatform/python-docs-samples/blob/main/cloud-sql/postgres/pgvector/notebooks/pgvector_gen_ai_demo.ipynb)
to store [items](https://www.kaggle.com/datasets/promptcloud/walmart-product-details-2020) and use slack as ui.
## Installation

#### 1. Clone the repository

```bash
git clone git@github.com:alanahmet/gcs-slack-app.git
```

#### 2. Create a Python environment

Python 3.6 or higher using `venv` or `conda`. Using `venv`:

``` bash
cd gcs-slack-app
python3 -m venv env
source env/bin/activate
```

Using `conda`:
``` bash
cd gcs-slack-app
conda create -n langchain-env python=3.11
conda activate langchain-env
```

#### 3. Install the required dependencies
``` bash
pip install -r requirements.txt
```

#### 4. Set up the keys in a .env file

Now your Python environment is set up, and you can proceed with running the experiments.

Tutorial slack app https://github.com/daveebbelaar/langchain-experiments
