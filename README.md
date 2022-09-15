# DataQA

![alt text](https://dataqa.ai/static/images/logo-violet.png "DataQA")

DataQA is a tool to perform AI model quality assessment (QA) using an interactive app that can be shared with technical and non-technical members of your team.

TODO: Add a gif.

The official documentation page is at: [docs.dataqa.ai]().

# Installation

`pip install dataqa`

# Quick start

## Step 1: create an account

Go to [https://app.dataqa.ai/](https://app.dataqa.ai/login) and follow the steps to create your first project. Once your account and your first project have been created, you will see a screen such as this one:

TODO: Add screenshot of the screen with the publish string

You will need this key later in order to be able to create your first QA app. You can always come back to this page to find it.

## Step 2: Publish your data

Creating your first shareable QA app is as simple as this:

```python
import pandas as pd
from dataqa.publish import DataQA
dataqa = DataQA()
dataqa.login()
# Prompt username and password
df = pd.DataFrame([[1, "Laptop", 1600], [2, "Mouse", 10]], columns=["id", "product", "price"])
dataqa.publish(PROJECT_ID, df)
```

The `PROJECT_ID` is the hash string on the dataqa project page.

## Step 3: Use the UI to explore your data

TODO: add screenshot or GIF

