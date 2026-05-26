# flake8: noqa
import os
from typing import Any, Literal, Optional

import mlflow
from dotenv import load_dotenv
from mlflow.entities.model_registry import (
    PromptModelConfig,
    PromptVersion,
)
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.store.entities.paged_list import PagedList
from pydantic import BaseModel

load_dotenv()

# os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_ADMIN_USER")
# os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_ADMIN_PASSWORD")

uri = os.getenv("MLFLOW_TRACKING_URI")


mlflow.set_tracking_uri(uri)


class RegisterPrompt(BaseModel):
    name: str
    template: str | list[dict[str, Any]]
    commit_message: str | None = None
    tags: dict[str, str] | None = None
    response_format: type[BaseModel] | dict[str, Any] | None = None
    model_settings: PromptModelConfig | dict[str, Any] | None = None


class LoadPrompt(BaseModel):
    name_or_uri: str
    version: str | int | None = None
    allow_missing: bool = False
    link_to_model: bool = True
    model_id: str | None = None
    cache_ttl_seconds: float | None = None


class SearchPrompt(BaseModel):
    filter_string: str | None = None
    max_results: int | None = None


class PromptManagement:
    def __init__(self):
        pass

    def set_experiment(self, experiment_name: str):

        mlflow.set_experiment(experiment_name)
        return f"Experiment set to '{experiment_name}'"

    def register_prompt(self, params: RegisterPrompt) -> PromptVersion:
        data = params.model_dump()
        data["model_config"] = data.pop("model_settings", None)

        mlflow.genai.register_prompt(**data)
        return f"Prompt '{params.name}' registered successfully"

    def load_prompt(self, params: LoadPrompt) -> PromptVersion:
        return mlflow.genai.load_prompt(**params.model_dump())

    def search_prompt(self, params: LoadPrompt) -> PagedList[Prompt]:
        return mlflow.genai.search_prompts(**params.model_dump())

    def delete_prompt(self, name: str, version: int) -> str:
        client = mlflow.MlflowClient()
        client.delete_prompt_version(name, version)
        return f"Prompt '{name}' deleted successfully"

    def set_prompt_tags(
        self,
        name: str,
        tag_operation: Literal["level", "version"] = "level",
        version: Optional[int] = None,
        key: str = None,
        value: str = None,
    ) -> PromptVersion:
        if tag_operation == "level":
            mlflow.genai.set_prompt_tag(name, key, value)
            return f"Prompt '{name}' tags updated successfully"
        else:
            mlflow.genai.set_prompt_version_tag(name, version, key, value)
            return f"Prompt '{name}' version '{version}' tags updated successfully"

    def get_prompt_tags(self, name: str) -> dict[str, str]:
        return mlflow.genai.get_prompt_tags(name)

    def delete_prompt_tags(
        self,
        name: str,
        tag_operation: Literal["level", "version"] = "level",
        version: Optional[int] = None,
        key: str = None,
    ):
        if tag_operation == "level":
            mlflow.genai.delete_prompt_tag(name, key)
            return f"Prompt '{name}' tags deleted successfully"
        else:
            mlflow.genai.delete_prompt_version_tag(name, version, key)
            return f"Prompt '{name}' version '{version}' tags deleted successfully"

    def set_model_config(
        self, name: str, model_config: PromptModelConfig, version: Optional[int] = None
    ):
        mlflow.genai.set_prompt_model_config(name, version, model_config)
        return f"Prompt '{name}' model config updated successfully"

    def delete_model_config(self, name: str, version: Optional[int] = None):
        mlflow.genai.delete_prompt_model_config(name, version)
        return f"Prompt '{name}' model config deleted successfully"

    def set_prompt_alias(self, name: str, alias: str, version: int) -> None:
        mlflow.genai.set_prompt_alias(name, alias, version)
        return f"Prompt '{name}' alias updated successfully with alias '{alias}' and version '{version}'"


if __name__ == "__main__":
    prompt_management = PromptManagement()
