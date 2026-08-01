"""
Hopsworks connection helper.

Single entry point: get_feature_store(). Reads credentials from
environment variables so nothing secret ever lives in code — in GitHub
Actions these come from repo secrets (see
.github/workflows/hourly_feature_pipeline.yml), locally from a .env file
or your shell environment.

Required env vars:
    HOPSWORKS_API_KEY   — required.
    HOPSWORKS_PROJECT    — optional. If your API key is scoped to a
                           single project, hopsworks.login() finds it
                           automatically and this can be omitted.
"""

import os

import hopsworks
from dotenv import load_dotenv

load_dotenv()

def get_feature_store():
    """
    Log into Hopsworks and return the project's feature store handle.

    Raises RuntimeError with a clear message if HOPSWORKS_API_KEY is
    missing, rather than letting the hopsworks library's own (less
    obvious) error surface first.
    """
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY environment variable is not set. "
            "In GitHub Actions this must be added as a repo secret and "
            "passed to the job's `env:` block; locally, export it or "
            "put it in a .env file that's loaded before this runs."
        )

    project_name = os.environ.get("HOPSWORKS_PROJECT")  

    login_kwargs = {"api_key_value": api_key}
    if project_name:
        login_kwargs["project"] = project_name

    project = hopsworks.login(**login_kwargs)
    return project.get_feature_store()
