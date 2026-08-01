
import os
from dotenv import load_dotenv
import hopsworks
load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME")
)

fs = project.get_feature_store()

fg = fs.get_feature_group(
    name="aqi_features",
    version=1
)

df = fg.read()

print("Rows:", len(df))
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())