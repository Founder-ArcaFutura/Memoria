import os
import sys

import pytest
from openai import AzureOpenAI

from memoria import Memoria
from memoria.core.providers import ProviderConfig
from tests.utils.test_utils import get_test_model

pytest.skip("Skipping interactive demo script.", allow_module_level=True)

# Load Azure OpenAI configuration from environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Skip tests if Azure OpenAI environment variables are not set
if not all(
    [
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT_NAME,
        AZURE_OPENAI_API_VERSION,
    ]
):
    pytest.skip(
        "Skipping Azure OpenAI tests because environment variables are not set.",
        allow_module_level=True,
    )

# Get test model from utility function
test_model = get_test_model(
    AZURE_OPENAI_DEPLOYMENT_NAME,
    {
        "gpt-4": "gpt-4",
        "gpt-3.5": "gpt-3.5-turbo",
        "gpt-35-turbo": "gpt-3.5-turbo",
    },
)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# Initialize Memoria with Azure provider configuration
provider_config = ProviderConfig(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    api_type="azure",
)

memoria = Memoria(
    provider_config=provider_config,
)
memoria.enable()
