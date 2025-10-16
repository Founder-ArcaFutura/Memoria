"""
Memory Agent - Async Pydantic-based conversation processing

This agent processes conversations and extracts structured information with
enhanced classification and conscious context detection.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import openai
from loguru import logger

from ..core.providers import (
    ProviderConfig,
    ProviderRegistry,
    ProviderSelection,
    ProviderType,
    ProviderUnavailableError,
    TaskRouteSpec,
)
from ..utils.embeddings import generate_embedding, vector_search_enabled
from ..utils.pydantic_models import (
    AgentPermissions,
    ConversationContext,
    MemoryClassification,
    MemoryImportanceLevel,
    ProcessedLongTermMemory,
)


class MemoryAgent:
    """
    Async Memory Agent for processing conversations with enhanced classification
    """

    _OPENAI_COMPATIBLE_TYPES = {
        ProviderType.OPENAI,
        ProviderType.AZURE,
        ProviderType.CUSTOM,
        ProviderType.OPENAI_COMPATIBLE,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        provider_config: ProviderConfig | None = None,
        permissions: AgentPermissions | None = None,
        provider_registry: ProviderRegistry | None = None,
        task_routes: Mapping[str, TaskRouteSpec] | None = None,
        conversation_manager: Any | None = None,
    ):
        """
        Initialize Memory Agent with LLM provider configuration

        Args:
            api_key: API key (deprecated, use provider_config)
            model: Model to use for structured output (defaults to 'gpt-4o-mini' if not specified)
            provider_config: Provider configuration for LLM client
        """
        if provider_config:
            # Use provider configuration to create clients
            self.client = provider_config.create_client()
            self.async_client = provider_config.create_async_client()
            # Use provided model, fallback to provider config model, then default to gpt-4o-mini
            self.model = model or provider_config.model or "gpt-4o-mini"
            logger.debug(f"Memory agent initialized with model: {self.model}")
            self.provider_config = provider_config
        else:
            # Backward compatibility: use api_key directly
            self.client = openai.OpenAI(api_key=api_key)
            self.async_client = openai.AsyncOpenAI(api_key=api_key)
            self.model = model or "gpt-4o-mini"
            self.provider_config = None

        # Determine if we're using a local/custom endpoint that might not support structured outputs
        self._supports_structured_outputs = self._detect_structured_output_support()
        self.permissions = permissions or AgentPermissions()
        self.vector_search_enabled = vector_search_enabled()
        self.provider_registry = provider_registry
        self.task_routes = dict(task_routes or {})
        self.conversation_manager = conversation_manager
        self._last_route: ProviderSelection | None = None
        self._memory_task_name = "memory_ingest"

    def set_conversation_manager(self, conversation_manager: Any) -> None:
        """Attach a conversation manager for routing telemetry."""

        self.conversation_manager = conversation_manager

    @property
    def last_route(self) -> ProviderSelection | None:
        """Return the most recent provider selection."""

        return self._last_route

    SYSTEM_PROMPT = """You are an advanced Memory Processing Agent responsible for analyzing conversations and extracting structured information with intelligent classification and conscious context detection.

Your primary functions:
1. **Intelligent Classification**: Categorize memories with enhanced classification system
2. **Conscious Context Detection**: Identify user context information for immediate promotion
3. **Entity Extraction**: Extract comprehensive entities and keywords
4. **Deduplication**: Identify and handle duplicate information
5. **Context Filtering**: Determine what should be stored vs filtered out

**ENHANCED CLASSIFICATION SYSTEM:**

**CONSCIOUS_INFO** (Auto-promote to short-term context):
- User's name, location, job, personal details
- Current projects, technologies they work with
- Preferences, work style, communication style
- Skills, expertise, learning goals
- Important personal context for AI interaction

**ESSENTIAL**:
- Core facts that define user's context
- Important preferences and opinions
- Key skills and knowledge areas
- Critical project information

**CONTEXTUAL**:
- Current work context
- Ongoing projects and goals
- Environmental setup and tools

**CONVERSATIONAL**:
- Regular discussions and questions
- Explanations and clarifications
- Problem-solving conversations

**REFERENCE**:
- Code examples and technical references
- Documentation and resources
- Learning materials

**PERSONAL**:
- Life events and personal information
- Relationships and social context
- Personal interests and hobbies

**IMPORTANCE LEVELS:**
- **CRITICAL**: Must never be lost
- **HIGH**: Very important for context
- **MEDIUM**: Useful to remember
- **LOW**: Nice to have context

**CONSCIOUS CONTEXT DETECTION:**
Mark is_user_context=True for:
- Personal identifiers (name, location, role)
- Work context (job, company, projects)
- Technical preferences (languages, tools, frameworks)
- Communication style and preferences
- Skills and expertise areas
- Learning goals and interests

Set promotion_eligible=True for memories that should be immediately available in short-term context for all future conversations.

**PROCESSING RULES:**
1. AVOID DUPLICATES: Check if similar information already exists
2. MERGE SIMILAR: Combine related information when appropriate
3. FILTER UNNECESSARY: Skip trivial greetings, acknowledgments
4. EXTRACT ENTITIES: Identify people, places, technologies, projects
5. ASSESS IMPORTANCE: Rate based on relevance to user context
6. FLAG USER CONTEXT: Mark information for conscious promotion

Focus on extracting information that would genuinely help provide better context and assistance in future conversations."""

    def _resolve_route_for_chat(
        self,
        chat_id: str,
        *,
        require_structured: bool,
        attempted: set[str] | None = None,
    ) -> ProviderSelection | None:
        """Select a provider for ``chat_id`` respecting prior failures."""

        if not self.provider_registry:
            self._last_route = None
            return None

        preferred_identifier = None
        if self.conversation_manager is not None:
            getter = getattr(self.conversation_manager, "get_last_model_route", None)
            if callable(getter):
                record = getter(chat_id, self._memory_task_name)
                if isinstance(record, Mapping) and record.get("success"):
                    provider = record.get("provider")
                    model = record.get("model")
                    if provider:
                        preferred_identifier = (
                            f"{provider}:{model}" if model else str(provider)
                        )

        selection = self.provider_registry.select(
            self._memory_task_name,
            require_structured=require_structured,
            preferred_provider=preferred_identifier,
            exclude_providers=attempted or (),
        )
        self._last_route = selection
        return selection

    def _record_route_usage(
        self,
        chat_id: str,
        selection: ProviderSelection | None,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record routing outcome in the conversation manager, if present."""

        if self.conversation_manager is None:
            return
        recorder = getattr(self.conversation_manager, "record_model_route", None)
        if not callable(recorder):
            return

        provider_name = "default"
        model_name = self.model
        fallback_used = False
        if selection is not None:
            provider_name = selection.provider_name
            model_name = selection.model or self.model
            fallback_used = bool(selection.fallback_used)

        try:
            recorder(
                chat_id,
                self._memory_task_name,
                provider_name,
                model_name,
                fallback_used,
                success=success,
                error=error,
            )
        except Exception:  # pragma: no cover - defensive logging guard
            logger.debug("Failed to record model routing telemetry", exc_info=True)

    def build_system_prompt(
        self,
        user_input: str,
        ai_output: str,
        context: ConversationContext | None = None,
        existing_memories: list[str] | None = None,
    ) -> tuple[str, str, str]:
        """Build conversation text, system prompt, and context info."""
        conversation_text = f"User: {user_input}\nAssistant: {ai_output}"

        system_prompt = self.SYSTEM_PROMPT
        if existing_memories:
            dedup_context = "\n\nEXISTING MEMORIES (for deduplication):\n" + "\n".join(
                existing_memories[:10]
            )
            system_prompt += dedup_context

        context_info = ""
        if context:
            context_info = f"""
CONVERSATION CONTEXT:
- Session: {context.session_id}
- Model: {context.model_used}
- User Projects: {', '.join(context.current_projects) if context.current_projects else 'None specified'}
- Relevant Skills: {', '.join(context.relevant_skills) if context.relevant_skills else 'None specified'}
- Topic Thread: {context.topic_thread or 'General conversation'}
"""

        return conversation_text, system_prompt, context_info

    async def process_with_structured_output(
        self,
        chat_id: str,
        system_prompt: str,
        conversation_text: str,
        context_info: str,
        selection: ProviderSelection | None = None,
    ) -> ProcessedLongTermMemory | None:
        """Process conversation using OpenAI structured outputs."""
        client = self.async_client
        model_to_use = self.model

        if selection is not None:
            if selection.provider_type not in self._OPENAI_COMPATIBLE_TYPES:
                raise ProviderUnavailableError(
                    f"Provider '{selection.provider_name}' is not OpenAI compatible"
                )
            client = selection.get_async_client()
            model_to_use = selection.model or model_to_use

        if client is None:
            raise ProviderUnavailableError(
                "Async client unavailable for structured output"
            )

        beta_api = getattr(client, "beta", None)
        if beta_api is None:
            raise ProviderUnavailableError(
                "Selected client does not expose structured output interface"
            )

        try:
            completion = await beta_api.chat.completions.parse(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Process this conversation for enhanced memory storage:\n\n{conversation_text}\n{context_info}",
                    },
                ],
                response_format=ProcessedLongTermMemory,
                temperature=0.1,
            )

            if completion.choices[0].message.refusal:
                logger.warning(
                    f"Memory processing refused for chat {chat_id}: {completion.choices[0].message.refusal}"
                )
                return self._create_empty_long_term_memory(
                    chat_id, "Processing refused for safety reasons"
                )

            processed_memory = completion.choices[0].message.parsed
            processed_memory.conversation_id = chat_id
            processed_memory.extraction_timestamp = datetime.now()
            self._maybe_attach_embedding(processed_memory)
            return processed_memory

        except ProviderUnavailableError:
            raise
        except Exception as e:
            logger.warning(
                f"Structured outputs failed for {chat_id}, falling back to manual parsing: {e}"
            )
            self._supports_structured_outputs = False
            return None

    async def process_conversation_async(
        self,
        chat_id: str,
        user_input: str,
        ai_output: str,
        context: ConversationContext | None = None,
        existing_memories: list[str] | None = None,
        relationship_context: dict[str, Any] | None = None,
    ) -> ProcessedLongTermMemory:
        """
        Async conversation processing with classification and conscious context detection

        Args:
            chat_id: Conversation ID
            user_input: User's input message
            ai_output: AI's response
            context: Additional conversation context
            existing_memories: list of existing memory summaries for deduplication

        Returns:
            Processed memory with classification and conscious flags
        """
        try:
            conversation_text, system_prompt, context_info = self.build_system_prompt(
                user_input, ai_output, context, existing_memories
            )

            processed_memory: ProcessedLongTermMemory | None = None
            selection: ProviderSelection | None = None
            attempted: set[str] = set()
            last_error: str | None = None

            if self._supports_structured_outputs:
                while True:
                    selection = self._resolve_route_for_chat(
                        chat_id,
                        require_structured=True,
                        attempted=attempted,
                    )

                    try:
                        processed_memory = await self.process_with_structured_output(
                            chat_id,
                            system_prompt,
                            conversation_text,
                            context_info,
                            selection=selection,
                        )
                        if processed_memory is not None:
                            break
                    except ProviderUnavailableError as exc:
                        last_error = str(exc)
                        if selection is None:
                            logger.warning(
                                f"Structured outputs unavailable for chat {chat_id}: {exc}"
                            )
                            self._supports_structured_outputs = False
                            break

                        logger.debug(
                            "Provider '{}' unavailable for structured output: {}",
                            selection.provider_name,
                            exc,
                        )
                        attempted.add(selection.provider_name.lower())
                        if self.provider_registry:
                            self.provider_registry.mark_unavailable(
                                selection.provider_name
                            )
                        self._record_route_usage(
                            chat_id,
                            selection,
                            success=False,
                            error=last_error,
                        )
                        continue
                    except Exception as e:
                        last_error = str(e)
                        logger.warning(f"Structured outputs failed for {chat_id}: {e}")
                        self._supports_structured_outputs = False
                        processed_memory = None
                        break

                    if selection is None:
                        break

            if processed_memory is None:
                if selection is None:
                    selection = self._resolve_route_for_chat(
                        chat_id,
                        require_structured=True,
                        attempted=attempted,
                    )
                try:
                    processed_memory = await self._process_with_fallback_parsing(
                        chat_id,
                        system_prompt,
                        conversation_text,
                        context_info,
                        selection=selection,
                    )
                except ProviderUnavailableError as exc:
                    last_error = str(exc)
                    logger.error(
                        f"Fallback parsing unavailable for chat {chat_id}: {exc}"
                    )
                    if selection is not None and self.provider_registry:
                        self.provider_registry.mark_unavailable(selection.provider_name)
                    self._record_route_usage(
                        chat_id,
                        selection,
                        success=False,
                        error=last_error,
                    )
                    return self._create_empty_long_term_memory(
                        chat_id, f"Processing failed: {last_error}"
                    )
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"Fallback parsing failed for chat {chat_id}: {e}")
                    self._record_route_usage(
                        chat_id,
                        selection,
                        success=False,
                        error=last_error,
                    )
                    return self._create_empty_long_term_memory(
                        chat_id, f"Processing failed: {last_error}"
                    )

            # Ensure the most recent selection is recorded for telemetry
            if selection is not None:
                self._last_route = selection
            else:
                self._last_route = None
            selection = self._last_route

            logger.debug(
                f"Processed conversation {chat_id}: "
                f"classification={processed_memory.classification}, "
                f"importance={processed_memory.importance}, "
                f"conscious_context={processed_memory.is_user_context}, "
                f"promotion_eligible={processed_memory.promotion_eligible}"
            )

            self._maybe_attach_embedding(processed_memory)

            if relationship_context:
                db_manager = (
                    relationship_context.get("db_manager")
                    if isinstance(relationship_context, dict)
                    else None
                )
                if db_manager:
                    try:
                        self.attach_related_memories(
                            processed_memory,
                            db_manager,
                            namespace=relationship_context.get("namespace", "default"),
                            limit=int(relationship_context.get("limit", 5) or 5),
                            privacy_floor=relationship_context.get(
                                "privacy_floor", -10.0
                            ),
                        )
                    except Exception as exc:
                        logger.debug(
                            f"Relationship detection failed for {chat_id}: {exc}"
                        )
            self._record_route_usage(chat_id, selection, success=True)
            return processed_memory

        except Exception as e:
            logger.error(f"Memory agent processing failed for {chat_id}: {e}")
            self._record_route_usage(
                chat_id, self._last_route, success=False, error=str(e)
            )
            return self._create_empty_long_term_memory(
                chat_id, f"Processing failed: {str(e)}"
            )

    def _create_empty_long_term_memory(
        self, chat_id: str, reason: str
    ) -> ProcessedLongTermMemory:
        """Create an empty long-term memory object for error cases"""
        return ProcessedLongTermMemory(
            content="Processing failed",
            summary="Processing failed",
            classification=MemoryClassification.CONVERSATIONAL,
            importance=MemoryImportanceLevel.LOW,
            conversation_id=chat_id,
            classification_reason=reason,
            confidence_score=0.0,
            extraction_timestamp=datetime.now(),
        )

    def _maybe_attach_embedding(self, memory: ProcessedLongTermMemory) -> None:
        """Attach an embedding to ``memory`` when vector search is enabled."""

        if not self.vector_search_enabled:
            return
        if getattr(memory, "embedding", None):
            return
        text = memory.summary or memory.content
        if not text:
            return
        try:
            memory.embedding = generate_embedding(text)
        except Exception as exc:  # pragma: no cover - deterministic fallback
            logger.debug(f"Failed to generate embedding for memory: {exc}")

    def attach_related_memories(
        self,
        memory: ProcessedLongTermMemory,
        db_manager: Any,
        *,
        namespace: str,
        limit: int = 5,
        privacy_floor: float | None = -10.0,
    ) -> list[dict[str, Any]]:
        """Populate ``memory.related_memories`` using database relationship helpers."""

        if not db_manager:
            return []

        finder = getattr(db_manager, "find_related_memory_candidates", None)
        if not callable(finder):
            return []

        anchors = list(memory.symbolic_anchors or [])
        keywords: list[str] = []
        if memory.keywords:
            keywords.extend(str(keyword) for keyword in memory.keywords if keyword)
        if memory.entities:
            keywords.extend(str(entity) for entity in memory.entities if entity)
        if memory.topic:
            keywords.append(str(memory.topic))
        if memory.summary:
            summary_tokens = [token for token in memory.summary.split() if token]
            keywords.extend(summary_tokens[:5])

        candidates = finder(
            namespace=namespace,
            symbolic_anchors=anchors,
            keywords=keywords,
            topic=memory.topic,
            exclude_ids=memory.related_memories or [],
            limit=limit,
            privacy_floor=privacy_floor,
        )

        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            candidate_id = (
                candidate.get("memory_id") if isinstance(candidate, dict) else None
            )
            if not candidate_id:
                continue
            if candidate_id in seen or candidate_id == memory.conversation_id:
                continue
            seen.add(candidate_id)
            ordered.append(candidate_id)
            if len(ordered) >= limit:
                break

        memory.related_memories = ordered
        return candidates

    def persist_relationship_links(
        self,
        memory_id: str,
        related_ids: Sequence[str],
        db_manager: Any,
        *,
        relation: str = "related",
    ) -> None:
        """Persist explicit relationship edges for a stored memory."""

        if not memory_id or not related_ids or not db_manager:
            return

        store_links = getattr(db_manager, "store_memory_links", None)
        if not callable(store_links):
            return

        try:
            store_links(memory_id, list(related_ids), relation=relation)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(f"Failed to persist relationship links for {memory_id}: {exc}")

    # === DEDUPLICATION & FILTERING METHODS ===

    async def detect_duplicates(
        self,
        new_memory: ProcessedLongTermMemory,
        existing_memories: list[ProcessedLongTermMemory],
        similarity_threshold: float = 0.8,
    ) -> str | None:
        """
        Detect if new memory is a duplicate of existing memories

        Args:
            new_memory: New memory to check
            existing_memories: list of existing memories to compare against
            similarity_threshold: Threshold for considering memories similar

        Returns:
            Memory ID of duplicate if found, None otherwise
        """
        # Simple text similarity check - could be enhanced with embeddings
        new_content = new_memory.content.lower().strip()
        new_summary = new_memory.summary.lower().strip()

        for existing in existing_memories:
            existing_content = existing.content.lower().strip()
            existing_summary = existing.summary.lower().strip()

            # Check content similarity
            content_similarity = self._calculate_similarity(
                new_content, existing_content
            )
            summary_similarity = self._calculate_similarity(
                new_summary, existing_summary
            )

            # Average similarity score
            avg_similarity = (content_similarity + summary_similarity) / 2

            if avg_similarity >= similarity_threshold:
                logger.info(
                    f"Duplicate detected: {avg_similarity:.2f} similarity with {existing.conversation_id}"
                )
                return existing.conversation_id

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Simple text similarity calculation using word overlap
        Could be enhanced with more sophisticated methods
        """
        if not text1 or not text2:
            return 0.0

        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _detect_structured_output_support(self) -> bool:
        """
        Detect if the current provider/endpoint supports OpenAI structured outputs

        Returns:
            True if structured outputs are likely supported, False otherwise
        """
        try:
            # Check if we have a provider config with custom base_url
            if self.provider_config and hasattr(self.provider_config, "base_url"):
                base_url = self.provider_config.base_url
                if base_url:
                    # Local/custom endpoints typically don't support beta features
                    if "localhost" in base_url or "127.0.0.1" in base_url:
                        logger.debug(
                            f"Detected local endpoint ({base_url}), disabling structured outputs"
                        )
                        return False
                    # Custom endpoints that aren't OpenAI
                    if "api.openai.com" not in base_url:
                        logger.debug(
                            f"Detected custom endpoint ({base_url}), disabling structured outputs"
                        )
                        return False

            # Check for Azure endpoints (they may or may not support beta features)
            if self.provider_config and hasattr(self.provider_config, "api_type"):
                if self.provider_config.api_type == "azure":
                    logger.debug(
                        "Detected Azure endpoint, enabling structured outputs (may need manual verification)"
                    )
                    return (
                        True  # Azure may support it, let it try and fallback if needed
                    )
                elif self.provider_config.api_type in ["custom", "openai_compatible"]:
                    logger.debug(
                        f"Detected {self.provider_config.api_type} endpoint, disabling structured outputs"
                    )
                    return False

            # Default: assume OpenAI endpoint supports structured outputs
            logger.debug("Assuming OpenAI endpoint, enabling structured outputs")
            return True

        except Exception as e:
            logger.debug(
                f"Error detecting structured output support: {e}, defaulting to enabled"
            )
            return True

    async def _process_with_fallback_parsing(
        self,
        chat_id: str,
        system_prompt: str,
        conversation_text: str,
        context_info: str,
        selection: ProviderSelection | None = None,
    ) -> ProcessedLongTermMemory:
        """
        Process conversation using regular chat completions with manual JSON parsing

        This method works with any OpenAI-compatible API that supports chat completions
        but doesn't support structured outputs (like Ollama, local models, etc.)
        """
        try:
            client = self.async_client
            model_to_use = self.model

            if selection is not None:
                if selection.provider_type not in self._OPENAI_COMPATIBLE_TYPES:
                    raise ProviderUnavailableError(
                        f"Provider '{selection.provider_name}' is not OpenAI compatible"
                    )
                client = selection.get_async_client()
                model_to_use = selection.model or model_to_use

            if client is None:
                raise ProviderUnavailableError(
                    "Async client unavailable for fallback parsing"
                )

            chat_api = getattr(client, "chat", None)
            if chat_api is None or not hasattr(chat_api, "completions"):
                raise ProviderUnavailableError(
                    "Selected client does not expose chat completion interface"
                )

            # Enhanced system prompt for JSON output
            json_system_prompt = (
                system_prompt
                + "\n\nIMPORTANT: You MUST respond with a valid JSON object that matches this exact schema:\n"
            )
            json_system_prompt += self._get_json_schema_prompt()
            json_system_prompt += "\n\nRespond ONLY with the JSON object, no additional text or formatting."

            # Call regular chat completions
            completion = await chat_api.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": json_system_prompt},
                    {
                        "role": "user",
                        "content": f"Process this conversation for enhanced memory storage:\n\n{conversation_text}\n{context_info}",
                    },
                ],
                temperature=0.1,  # Low temperature for consistent processing
                max_tokens=2000,  # Ensure enough tokens for full response
            )

            # Extract and parse JSON response
            response_text = completion.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from model")

            # Clean up response (remove markdown formatting if present)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Parse JSON
            try:
                parsed_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for {chat_id}: {e}")
                logger.debug(f"Raw response: {response_text}")
                return self._create_empty_long_term_memory(
                    chat_id, f"JSON parsing failed: {e}"
                )

            # Convert to ProcessedLongTermMemory object with validation and defaults
            processed_memory = self._create_memory_from_dict(parsed_data, chat_id)
            self._maybe_attach_embedding(processed_memory)

            logger.debug(
                f"Successfully parsed memory using fallback method for {chat_id}"
            )
            return processed_memory

        except Exception as e:
            logger.error(f"Fallback memory processing failed for {chat_id}: {e}")
            return self._create_empty_long_term_memory(
                chat_id, f"Fallback processing failed: {str(e)}"
            )

    def _get_json_schema_prompt(self) -> str:
        """
        Get JSON schema description for manual parsing
        """
        return """{
  "content": "string - The actual memory content",
  "summary": "string - Concise summary for search",
  "classification": "string - One of: essential, contextual, conversational, reference, personal, conscious-info",
  "importance": "string - One of: critical, high, medium, low",
  "topic": "string or null - Main topic/subject",
  "entities": ["array of strings - People, places, technologies mentioned"],
  "keywords": ["array of strings - Key terms for search"],
  "is_user_context": "boolean - Contains user personal info",
  "is_preference": "boolean - User preference/opinion",
  "is_skill_knowledge": "boolean - User's abilities/expertise",
  "is_current_project": "boolean - Current work context",
  "classification_reason": "string - Why this classification was chosen",
  "confidence_score": "number - AI confidence in extraction (0.0-1.0)",
  "promotion_eligible": "boolean - Should be promoted to short-term",
  "x_coord": "number or null - X-axis coordinate",
  "y_coord": "number or null - Y-axis coordinate",
  "z_coord": "number or null - Z-axis coordinate",
  "emotional_intensity": "number or null - Emotional intensity (0.0-1.0)",
  "symbolic_anchors": "array of strings or null - Symbolic location anchors"
}"""

    def _create_memory_from_dict(
        self, data: dict[str, Any], chat_id: str
    ) -> ProcessedLongTermMemory:
        """
        Create ProcessedLongTermMemory from dictionary with proper validation and defaults
        """
        try:
            # Import here to avoid circular imports
            from ..utils.pydantic_models import (
                MemoryClassification,
                MemoryImportanceLevel,
            )

            # Validate and convert classification
            classification_str = (
                data.get("classification", "conversational").lower().replace("_", "-")
            )
            try:
                classification = MemoryClassification(classification_str)
            except ValueError:
                logger.warning(
                    f"Invalid classification '{classification_str}', using 'conversational'"
                )
                classification = MemoryClassification.CONVERSATIONAL

            # Validate and convert importance
            importance_str = data.get("importance", "medium").lower()
            try:
                importance = MemoryImportanceLevel(importance_str)
            except ValueError:
                logger.warning(f"Invalid importance '{importance_str}', using 'medium'")
                importance = MemoryImportanceLevel.MEDIUM

            # Create memory object with proper validation
            processed_memory = ProcessedLongTermMemory(
                content=data.get("content", "No content extracted"),
                summary=data.get("summary", "No summary available"),
                classification=classification,
                importance=importance,
                topic=data.get("topic"),
                entities=data.get("entities", []),
                keywords=data.get("keywords", []),
                is_user_context=bool(data.get("is_user_context", False)),
                is_preference=bool(data.get("is_preference", False)),
                is_skill_knowledge=bool(data.get("is_skill_knowledge", False)),
                is_current_project=bool(data.get("is_current_project", False)),
                x_coord=(
                    float(data.get("x_coord"))
                    if data.get("x_coord") is not None
                    else None
                ),
                y_coord=(
                    float(data.get("y_coord"))
                    if data.get("y_coord") is not None
                    else None
                ),
                z_coord=(
                    float(data.get("z_coord"))
                    if data.get("z_coord") is not None
                    else None
                ),
                emotional_intensity=(
                    float(data.get("emotional_intensity"))
                    if data.get("emotional_intensity") is not None
                    else None
                ),
                symbolic_anchors=(
                    data.get("symbolic_anchors")
                    or (
                        [data.get("symbolic_anchor")]
                        if data.get("symbolic_anchor")
                        else None
                    )
                ),
                conversation_id=chat_id,
                confidence_score=float(data.get("confidence_score", 0.7)),
                classification_reason=data.get(
                    "classification_reason", "Extracted via fallback parsing"
                ),
                promotion_eligible=bool(data.get("promotion_eligible", False)),
                extraction_timestamp=datetime.now(),
            )

            return processed_memory

        except Exception as e:
            logger.error(f"Error creating memory from dict: {e}")
            return self._create_empty_long_term_memory(
                chat_id, f"Memory creation failed: {str(e)}"
            )

    def should_filter_memory(
        self, memory: ProcessedLongTermMemory, filters: dict[str, Any] | None = None
    ) -> bool:
        """
        Determine if memory should be filtered out

        Args:
            memory: Memory to check
            filters: Optional filtering criteria

        Returns:
            True if memory should be filtered out, False otherwise
        """
        if not filters:
            return False

        # Classification filter
        if "exclude_classifications" in filters:
            if memory.classification in filters["exclude_classifications"]:
                return True

        # Importance filter
        if "min_importance" in filters:
            importance_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}

            min_level = importance_map.get(filters["min_importance"], 1)
            memory_level = importance_map.get(memory.importance, 1)

            if memory_level < min_level:
                return True

        # Confidence filter
        if "min_confidence" in filters:
            if memory.confidence_score < filters["min_confidence"]:
                return True

        # Content filters
        if "exclude_keywords" in filters:
            content_lower = memory.content.lower()
            if any(
                keyword.lower() in content_lower
                for keyword in filters["exclude_keywords"]
            ):
                return True

        # Length filter
        if "min_content_length" in filters:
            if len(memory.content.strip()) < filters["min_content_length"]:
                return True

        return False
