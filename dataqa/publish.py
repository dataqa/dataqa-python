from getpass import getpass
import json
import requests
from typing import Optional

import pandas as pd

from dataqa.column_mapping import ColumnMapping
from dataqa.infer_schema import infer_schema, validate_schema


API_URL = "https://django.dataqa.ai"
DOMAIN = "https://app.dataqa.ai"


class DataQA:
    def __init__(self, domain: str = DOMAIN, api_url: str = API_URL):
        self.domain = domain
        self.api_url = api_url

    def login(self):
        username = input("Username: ")
        print("")
        password = getpass(prompt="Password: ")
        response = requests.post(
            self.api_url + "/api/token/",
            headers={"Content-type": "application/json"},
            data=json.dumps({"username": username, "password": password}),
        )

        self.auth_token = response.json()["token"]

    def create_release(self, project_id: str, column_mapping: list[dict]) -> str:
        response = requests.post(
            self.api_url + "/api/v1/release/",
            headers={
                "Authorization": f"Token {self.auth_token}",
            },
            json={"project": project_id, "column_mapping": column_mapping},
        )

        release_id = response.json()["id"]
        return release_id

    def publish_data(self, df: pd.DataFrame, release_id: str):
        row_list = df.values.tolist()
        _ = requests.post(
            self.api_url + "/api/v1/releasedata/",
            headers={
                "Authorization": f"Token {self.auth_token}",
            },
            json={"release": release_id, "published_data": row_list},
        )

    def publish(
        self,
        project_id: str,
        df: pd.DataFrame,
        column_mapping: Optional[ColumnMapping] = None,
    ):
        if len(df) == 0:
            raise Exception("Empty dataframe.")

        if not column_mapping:
            column_mapping = infer_schema(df)

        schema, df = validate_schema(df, column_mapping)

        release_id = self.create_release(project_id, schema)
        self.publish_data(df, release_id)

        return f"{self.domain}/release/{release_id}"
