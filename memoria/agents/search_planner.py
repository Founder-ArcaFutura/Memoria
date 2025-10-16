"""Search planning logic for MemorySearchEngine."""

import json
import threading
import time
from typing import Any

from loguru import logger

from ..core.providers import ProviderType, ProviderUnavailableError
from ..utils.pydantic_models import AgentPermissions, MemorySearchQuery
from .search_prompts import SYSTEM_PROMPT, get_search_query_json_schema


class SearchPlanner:
    """Plan search strategies using an LLM."""

    def __init__(
        self,
        client,
        model: str,
        provider_config=None,
        permissions: AgentPermissions | None = None,
        provider_type: ProviderType | None = None,
    ):
        self.client = client
        self.model = model
        self.provider_config = provider_config
        self.permissions = permissions or AgentPermissions()
        self.provider_type = provider_type
        self._supports_structured_outputs = self._detect_structured_output_support()
        self._query_cache: dict[str, Any] = {}
        self._cache_ttl = 300
        self._cache_lock = threading.Lock()

    def set_provider_type(self, provider_type: ProviderType | None) -> None:
        """Update provider type and recompute structured output support."""

        self.provider_type = provider_type
        self._supports_structured_outputs = self._detect_structured_output_support()

    def plan_search(self, query: str, context: str | None = None) -> MemorySearchQuery:
        """Plan search strategy for a user query with caching."""
        try:
            cache_key = f"{query}|{context or ''}"
            with self._cache_lock:
                if cache_key in self._query_cache:
                    cached_result, timestamp = self._query_cache[cache_key]
                    if time.time() - timestamp < self._cache_ttl:
                        logger.debug(f"Using cached search plan for: {query}")
                        return cached_result

            prompt = f"User query: {query}"
            if context:
                prompt += f"\nAdditional context: {context}"

            search_query = None
            if self._supports_structured_outputs:
                try:
                    completion = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        response_format=MemorySearchQuery,
                        temperature=0.1,
                    )
                    if completion.choices[0].message.refusal:
                        logger.warning(
                            f"Search planning refused: {completion.choices[0].message.refusal}"
                        )
                        return self._create_fallback_query(query)
                    search_query = completion.choices[0].message.parsed
                except Exception as e:
                    logger.warning(
                        f"Structured outputs failed for search planning, falling back to manual parsing: {e}"
                    )
                    self._supports_structured_outputs = False
                    search_query = None

            if search_query is None:
                search_query = self._plan_search_with_fallback_parsing(query)

            with self._cache_lock:
                self._query_cache[cache_key] = (search_query, time.time())
                self._cleanup_cache()

            logger.debug(
                f"Planned search for query '{query}': intent='{search_query.intent}', strategies={search_query.search_strategy}"
            )
            return search_query
        except Exception as e:
            logger.error(f"Search planning failed: {e}")
            return self._create_fallback_query(query)

    def _plan_search_with_fallback_parsing(self, query: str) -> MemorySearchQuery:
        """Plan search strategy using regular chat completions with manual JSON parsing."""
        try:
            prompt = f"User query: {query}"
            json_system_prompt = (
                SYSTEM_PROMPT
                + "\n\nIMPORTANT: You MUST respond with a valid JSON object that matches this exact schema:\n"
            )
            json_system_prompt += get_search_query_json_schema()
            json_system_prompt += "\n\nRespond ONLY with the JSON object, no additional text or formatting."

            response_text = self._generate_completion_text(
                system_prompt=json_system_prompt,
                user_prompt=prompt,
            )

            if not response_text:
                raise ValueError("Empty response from model")

            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            try:
                parsed_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for search planning: {e}")
                logger.debug(f"Raw response: {response_text}")
                return self._create_fallback_query(query)

            search_query = self._create_search_query_from_dict(parsed_data, query)
            logger.debug("Successfully parsed search query using fallback method")
            return search_query
        except ProviderUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Fallback search planning failed: {e}")
            return self._create_fallback_query(query)

    def _generate_completion_text(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return raw completion text from the configured provider."""

        try:
            if self.provider_type == ProviderType.ANTHROPIC:
                response = self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=1000,
                    temperature=0.1,
                )
                text_parts: list[str] = []
                content = getattr(response, "content", []) or []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    elif hasattr(part, "text"):
                        text_parts.append(getattr(part, "text", ""))
                if not text_parts and hasattr(response, "completion"):
                    text_parts.append(getattr(response, "completion", ""))
                text = "".join(text_parts).strip()
                if not text:
                    raise ProviderUnavailableError(
                        "Anthropic response missing text content"
                    )
                return text

            if self.provider_type == ProviderType.GOOGLE_GEMINI:
                response = self.client.generate_content(
                    [system_prompt, user_prompt],
                )
                if hasattr(response, "text") and response.text:
                    return response.text
                candidates = getattr(response, "candidates", []) or []
                for candidate in candidates:
                    content = getattr(candidate, "content", []) or []
                    for part in content:
                        if isinstance(part, dict) and part.get("text"):
                            return str(part.get("text"))
                        if hasattr(part, "text") and part.text:
                            return str(part.text)
                raise ProviderUnavailableError("Gemini response missing text content")

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            message = completion.choices[0].message
            text = getattr(message, "content", None) or getattr(message, "text", "")
            if not text:
                raise ValueError("Empty response from model")
            return str(text)
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(str(exc)) from exc

    def _create_search_query_from_dict(
        self, data: dict[str, Any], original_query: str
    ) -> MemorySearchQuery:
        """Create MemorySearchQuery from dictionary with validation and defaults."""
        try:
            from ..utils.pydantic_models import MemoryCategoryType

            category_filters = []
            raw_categories = data.get("category_filters", [])
            if isinstance(raw_categories, list):
                for cat_str in raw_categories:
                    try:
                        category = MemoryCategoryType(cat_str.lower())
                        category_filters.append(category)
                    except ValueError:
                        logger.debug(f"Invalid category filter '{cat_str}', skipping")

            search_query = MemorySearchQuery(
                query_text=data.get("query_text", original_query),
                intent=data.get("intent", "General search (fallback)"),
                entity_filters=data.get("entity_filters", []),
                category_filters=category_filters,
                time_range=data.get("time_range"),
                min_importance=max(
                    0.0, min(1.0, float(data.get("min_importance", 0.0)))
                ),
                search_strategy=data.get("search_strategy", ["keyword_search"]),
                expected_result_types=data.get("expected_result_types", ["any"]),
            )
            return search_query
        except Exception as e:
            logger.error(f"Error creating search query from dict: {e}")
            return self._create_fallback_query(original_query)

    def _create_fallback_query(self, query: str) -> MemorySearchQuery:
        return MemorySearchQuery(
            query_text=query,
            intent="General search (fallback)",
            entity_filters=[word for word in query.split() if len(word) > 2],
            search_strategy=["keyword_search", "general_search"],
            expected_result_types=["any"],
        )

    def _cleanup_cache(self):
        current_time = time.time()
        expired = [
            key
            for key, (_, timestamp) in self._query_cache.items()
            if current_time - timestamp >= self._cache_ttl
        ]
        for key in expired:
            del self._query_cache[key]

    def _detect_structured_output_support(self) -> bool:
        try:
            if self.provider_type and self.provider_type not in {
                ProviderType.OPENAI,
                ProviderType.AZURE,
                ProviderType.CUSTOM,
                ProviderType.OPENAI_COMPATIBLE,
            }:
                logger.debug(
                    "Structured outputs disabled for provider type {}",
                    self.provider_type,
                )
                return False

            if self.provider_config and hasattr(self.provider_config, "base_url"):
                base_url = self.provider_config.base_url
                if base_url:
                    if "localhost" in base_url or "127.0.0.1" in base_url:
                        logger.debug(
                            f"Detected local endpoint ({base_url}), disabling structured outputs"
                        )
                        return False
                    if "api.openai.com" not in base_url:
                        logger.debug(
                            f"Detected custom endpoint ({base_url}), disabling structured outputs"
                        )
                        return False
            if self.provider_config and hasattr(self.provider_config, "api_type"):
                if self.provider_config.api_type == "azure":
                    return self._test_azure_structured_outputs_support()
                elif self.provider_config.api_type in [
                    "custom",
                    "openai_compatible",
                ]:
                    logger.debug(
                        f"Detected {self.provider_config.api_type} endpoint, disabling structured outputs"
                    )
                    return False
            logger.debug("Assuming OpenAI endpoint, enabling structured outputs")
            return True
        except Exception as e:
            logger.debug(
                f"Error detecting structured output support: {e}, defaulting to enabled"
            )
            return True

    def _test_azure_structured_outputs_support(self) -> bool:
        try:
            from pydantic import BaseModel

            class TestModel(BaseModel):
                test_field: str

            test_response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": "Say hello"}],
                response_format=TestModel,
                max_tokens=10,
                temperature=0,
            )
            if (
                test_response
                and hasattr(test_response, "choices")
                and test_response.choices
            ):
                logger.debug(
                    "Azure endpoint supports structured outputs - test successful"
                )
                return True
            logger.debug(
                "Azure endpoint structured outputs test failed - response invalid"
            )
            return False
        except Exception as e:
            logger.debug(f"Azure endpoint doesn't support structured outputs: {e}")
            return False
