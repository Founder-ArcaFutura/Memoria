"""
Conversation Manager for Stateless LLM SDK Integration

This module provides conversation tracking and context management for stateless LLM SDKs
like OpenAI, Anthropic, etc. It bridges the gap between memoria's stateful memory system
and stateless LLM API calls by maintaining conversation history and context.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from memoria.core.context_injection import (
    _extract_latest_user_input,
    _select_context_rows,
    format_context_prompt,
)


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation"""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """Represents an active conversation session"""

    session_id: str
    messages: list[ConversationMessage] = field(default_factory=list)
    context_injected: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: dict[str, Any] = None):
        """Add a message to the conversation"""
        message = ConversationMessage(
            role=role, content=content, metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_accessed = datetime.now()

    def get_history_messages(self, limit: int = 10) -> list[dict[str, str]]:
        """Get conversation history in OpenAI message format"""
        # Get recent messages (excluding system messages)
        user_assistant_messages = [
            msg for msg in self.messages if msg.role in ["user", "assistant"]
        ]

        # Limit to recent messages to prevent context overflow
        recent_messages = (
            user_assistant_messages[-limit:] if limit > 0 else user_assistant_messages
        )

        return [{"role": msg.role, "content": msg.content} for msg in recent_messages]


def _normalize_user_message_content(content: Any) -> str:
    """Return user-authored content collapsed into a plain string."""

    normalized = _extract_latest_user_input([{"role": "user", "content": content}])
    if normalized:
        return normalized
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


class ConversationManager:
    """
    Manages conversation sessions for stateless LLM integrations.

    This class provides:
    - Session-based conversation tracking
    - Context injection with conversation history
    - Automatic session cleanup
    - Support for both conscious_ingest and auto_ingest modes
    """

    def __init__(
        self,
        max_sessions: int = 100,
        session_timeout_minutes: int = 60,
        max_history_per_session: int = 20,
    ):
        """
        Initialize ConversationManager

        Args:
            max_sessions: Maximum number of active sessions
            session_timeout_minutes: Session timeout in minutes
            max_history_per_session: Maximum messages to keep per session
        """
        self.max_sessions = max_sessions
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_history_per_session = max_history_per_session

        # Active conversation sessions
        self.sessions: dict[str, ConversationSession] = {}

        logger.info(
            f"ConversationManager initialized: max_sessions={max_sessions}, "
            f"timeout={session_timeout_minutes}min, max_history={max_history_per_session}"
        )

    def get_or_create_session(self, session_id: str = None) -> ConversationSession:
        """
        Get existing session or create new one

        Args:
            session_id: Optional session ID. If None, generates new one.

        Returns:
            ConversationSession instance
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Clean up expired sessions first
        self._cleanup_expired_sessions()

        # Get existing session or create new one
        if session_id not in self.sessions:
            if len(self.sessions) >= self.max_sessions:
                # Remove oldest session to make room
                oldest_session_id = min(
                    self.sessions.keys(),
                    key=lambda sid: self.sessions[sid].last_accessed,
                )
                del self.sessions[oldest_session_id]
                logger.debug(f"Removed oldest session {oldest_session_id} to make room")

            self.sessions[session_id] = ConversationSession(session_id=session_id)
            logger.debug(f"Created new conversation session: {session_id}")
        else:
            # Update last accessed time
            self.sessions[session_id].last_accessed = datetime.now()

        return self.sessions[session_id]

    def add_user_message(
        self, session_id: str, content: str, metadata: dict[str, Any] = None
    ):
        """Add user message to conversation session"""
        session = self.get_or_create_session(session_id)
        normalized_content = _normalize_user_message_content(content)
        session.add_message("user", normalized_content, metadata)

        # Limit history to prevent memory bloat
        if len(session.messages) > self.max_history_per_session:
            # Keep system messages and recent messages
            system_messages = [msg for msg in session.messages if msg.role == "system"]
            other_messages = [msg for msg in session.messages if msg.role != "system"]

            # Keep recent non-system messages
            recent_messages = other_messages[
                -(self.max_history_per_session - len(system_messages)) :
            ]
            session.messages = system_messages + recent_messages

            logger.debug(f"Trimmed conversation history for session {session_id}")

    def add_assistant_message(
        self, session_id: str, content: str, metadata: dict[str, Any] = None
    ):
        """Add assistant message to conversation session"""
        session = self.get_or_create_session(session_id)
        session.add_message("assistant", content, metadata)

    def record_model_route(
        self,
        session_id: str,
        task: str,
        provider_name: str,
        model_name: str | None,
        fallback_used: bool,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Track the provider/model used for a specific task."""

        if not session_id:
            return

        session = self.get_or_create_session(session_id)
        routes = session.metadata.setdefault("model_routes", {})
        routes[task] = {
            "provider": provider_name,
            "model": model_name,
            "fallback_used": bool(fallback_used),
            "success": bool(success),
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }

    def get_last_model_route(self, session_id: str, task: str) -> dict[str, Any] | None:
        """Return the most recent routing metadata for ``task``."""

        if not session_id:
            return None
        session = self.sessions.get(session_id)
        if session is None:
            return None
        routes = session.metadata.get("model_routes")
        if isinstance(routes, dict):
            record = routes.get(task)
            if isinstance(record, dict):
                return dict(record)
        return None

    def inject_context_with_history(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        memoria_instance,
        mode: str = "conscious",
    ) -> list[dict[str, str]]:
        """
        Inject context and conversation history into messages

        Args:
            session_id: Conversation session ID
            messages: Original messages from API call
            memoria_instance: Memoria instance for context retrieval
            mode: Context injection mode ("conscious" or "auto")

        Returns:
            Modified messages with context and history injected
        """
        try:
            session = self.get_or_create_session(session_id)

            # Extract user input from current messages
            user_input = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_input = _normalize_user_message_content(msg.get("content", ""))
                    break

            # Add current user message to session history
            if user_input:
                self.add_user_message(session_id, user_input)

            # Build context based on mode
            context_prompt = ""

            if mode == "conscious":
                # Conscious mode: Always inject short-term memory context
                # (Not just once - this fixes the original bug)
                plan = None
                context: list[dict[str, Any]]
                if hasattr(memoria_instance, "prepare_context_window"):
                    try:
                        context, plan = memoria_instance.prepare_context_window(
                            "conscious",
                            user_input,
                        )
                    except TypeError:  # pragma: no cover - compatibility shim
                        context, plan = memoria_instance.prepare_context_window("conscious", user_input)  # type: ignore[misc]
                else:
                    context = memoria_instance._get_conscious_context()
                if context:
                    filtered_context = _select_context_rows(context, plan)
                    context_prompt = format_context_prompt(
                        "conscious",
                        filtered_context,
                        plan=plan,
                    )
                    logger.debug(
                        "Injected conscious context with %s items for session %s",
                        len(filtered_context),
                        session_id,
                    )

            elif mode == "auto":
                # Auto mode: Search long-term memory database for relevant context
                logger.debug(
                    f"Auto-ingest: Processing user input for long-term memory search: '{user_input[:50]}...'"
                )
                plan = None
                context: list[dict[str, Any]]
                if hasattr(memoria_instance, "prepare_context_window"):
                    try:
                        context, plan = memoria_instance.prepare_context_window(
                            "auto",
                            user_input,
                        )
                    except TypeError:  # pragma: no cover - compatibility shim
                        context, plan = memoria_instance.prepare_context_window("auto", user_input)  # type: ignore[misc]
                else:
                    context = (
                        memoria_instance._get_auto_ingest_context(user_input)
                        if user_input
                        else []
                    )
                if context:
                    filtered_context = _select_context_rows(context, plan)
                    context_prompt = format_context_prompt(
                        "auto",
                        filtered_context,
                        plan=plan,
                    )
                    logger.debug(
                        "Auto-ingest: Successfully injected long-term memory context with %s items for session %s",
                        len(filtered_context),
                        session_id,
                    )
                else:
                    logger.debug(
                        f"Auto-ingest: No relevant memories found in long-term database for query '{user_input[:50]}...' in session {session_id}"
                    )

            # Get conversation history
            history_messages = session.get_history_messages(limit=10)

            # Build enhanced messages with context and history
            enhanced_messages = []

            # Add system message with context if we have any
            system_content = ""

            if context_prompt:
                system_content += context_prompt

            # Add conversation history if available (excluding current message)
            if len(history_messages) > 1:  # More than just current message
                previous_messages = history_messages[:-1]  # Exclude current message
                if previous_messages:
                    system_content += "\n--- Conversation History ---\n"
                    for msg in previous_messages:
                        role_label = "You" if msg["role"] == "assistant" else "User"
                        system_content += f"{role_label}: {msg['content']}\n"
                    system_content += "--- End History ---\n"
                    logger.debug(
                        f"Added {len(previous_messages)} history messages for session {session_id}"
                    )

            # Find existing system message or create new one
            has_system_message = False
            for msg in messages:
                if msg.get("role") == "system":
                    # Prepend our context to existing system message
                    if system_content:
                        msg["content"] = system_content + "\n" + msg.get("content", "")
                    enhanced_messages.append(msg)
                    has_system_message = True
                else:
                    enhanced_messages.append(msg)

            # If no system message exists and we have context/history, add one
            if not has_system_message and system_content:
                enhanced_messages.insert(
                    0, {"role": "system", "content": system_content}
                )

            logger.debug(
                f"Enhanced messages for session {session_id}: context={'yes' if context_prompt else 'no'}, "
                f"history={'yes' if len(history_messages) > 1 else 'no'}"
            )

            return enhanced_messages

        except Exception as e:
            logger.error(
                f"Failed to inject context with history for session {session_id}: {e}"
            )
            return messages

    def record_response(
        self, session_id: str, response: str, metadata: dict[str, Any] = None
    ):
        """Record AI response in conversation history"""
        try:
            self.add_assistant_message(session_id, response, metadata)
            logger.debug(f"Recorded AI response for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to record response for session {session_id}: {e}")

    def get_session_stats(self) -> dict[str, Any]:
        """Get conversation manager statistics"""
        return {
            "active_sessions": len(self.sessions),
            "max_sessions": self.max_sessions,
            "session_timeout_minutes": self.session_timeout.total_seconds() / 60,
            "max_history_per_session": self.max_history_per_session,
            "sessions": {
                session_id: {
                    "message_count": len(session.messages),
                    "created_at": session.created_at.isoformat(),
                    "last_accessed": session.last_accessed.isoformat(),
                    "context_injected": session.context_injected,
                }
                for session_id, session in self.sessions.items()
            },
        }

    def clear_session(self, session_id: str):
        """Clear a specific conversation session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared conversation session: {session_id}")

    def clear_all_sessions(self):
        """Clear all conversation sessions"""
        session_count = len(self.sessions)
        self.sessions.clear()
        logger.info(f"Cleared all {session_count} conversation sessions")

    def _cleanup_expired_sessions(self):
        """Remove expired conversation sessions"""
        now = datetime.now()
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if now - session.last_accessed > self.session_timeout
        ]

        for session_id in expired_sessions:
            del self.sessions[session_id]

        if expired_sessions:
            logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
