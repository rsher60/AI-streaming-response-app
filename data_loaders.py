import os
import warnings

import numpy as np
import pandas as pd
import openai
import weaviate
import weaviate.classes as wvc

from dotenv import load_dotenv, dotenv_values
from langchain_community.embeddings import OpenAIEmbeddings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

load_dotenv()

import pandas as pd
import numpy as np
import json



df = pd.read_csv('./data/cleaned_recipe_df.csv')

print(df.head())



config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]
weaviate_api_key = config["WCS_API_KEY"]
weaviate_url = config["WCS_URL"]

#create a weaviate client

headers = {
    "X-OpenAI-Api-Key": openai.api_key,
}



import weaviate.classes as wvc
with weaviate.connect_to_wcs(
    cluster_url=weaviate_url,  # Replace with your Weaviate Cloud URL
    auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),  # Replace with your Weaviate Cloud key
    headers=headers,
    skip_init_checks=True
) as client:  # Use this context manager to ensure the connection is closed
    print(client.is_ready())


from weaviate.classes.config import Configure

client.connect()
recipes = client.collections.create(
        "Recipe",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
        properties=[
        wvc.config.Property(
            name="title",
            data_type=wvc.config.DataType.TEXT,
        ),
        wvc.config.Property(
            name="ingredients",
            data_type=wvc.config.DataType.TEXT,
        ),
        wvc.config.Property(
            name="directions",
            data_type=wvc.config.DataType.TEXT,
        ),
        wvc.config.Property(
            name="directions_text",
            data_type=wvc.config.DataType.TEXT,
        ),

    ],
    # Configure the vector index
    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(  # Or `flat` or `dynamic`
        distance_metric=wvc.config.VectorDistances.COSINE,
        quantizer=wvc.config.Configure.VectorIndex.Quantizer.bq(),
    ),
)


recipe_objs = list()
for i, d in df.iterrows():
    recipe_objs.append(wvc.data.DataObject(
        properties={
            "title": d["title"],
            "ingredients": d["ingredients"],
            "directions": d["directions"],
            "directions_text" : d["directions_text"]
        },
    ))

client.connect()
Recipe = client.collections.get("Recipe")
Recipe.data.insert_many(recipe_objs)






