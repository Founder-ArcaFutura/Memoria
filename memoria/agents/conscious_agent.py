"""
Conscious Agent for User Context Management

This agent copies conscious-info labeled memories from long-term memory
directly to short-term memory for immediate context availability.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from ..conscious.constants import CONSCIOUS_CONTEXT_CATEGORY
from ..utils.pydantic_models import AgentPermissions


# NOTE: Class name corrected from a previous typo for clarity and consistency.
class ConsciousAgent:
    """
    Agent that copies conscious-info labeled memories from long-term memory
    directly to short-term memory for immediate context availability.

    Runs once at program startup when conscious_ingest=True.
    """

    def __init__(
        self,
        permissions: AgentPermissions | None = None,
        use_heuristics: bool = False,
        heuristic_config: dict[str, Any] | None = None,
    ):
        """Initialize the conscious agent"""
        self.context_initialized = False
        self.permissions = permissions or AgentPermissions()
        self.use_heuristics = use_heuristics

        default_config = {
            "min_summary_chars": 24,
            "min_searchable_chars": 64,
            "max_age_minutes": 60 * 24,
            "min_match_chars": 24,
            "match_snippet_chars": 120,
            "candidate_scan_limit": 25,
        }
        self.heuristic_config = default_config.copy()
        if heuristic_config:
            self.heuristic_config.update(heuristic_config)

    async def run_conscious_ingest(
        self, db_manager, namespace: str = "default"
    ) -> bool:
        """
        Run conscious context ingestion once at program startup

        Copies all conscious-info labeled memories from long-term memory
        directly to short-term memory as permanent context

        Args:
            db_manager: Database manager instance
            namespace: Memory namespace

        Returns:
            True if memories were copied, False otherwise
        """
        if not getattr(db_manager, "enable_short_term", True):
            logger.info("Short-term memory disabled; skipping conscious ingest")
            return False
        try:
            # Get all conscious-info labeled memories
            conscious_memories = await self._get_conscious_memories(
                db_manager, namespace
            )

            if not conscious_memories:
                logger.info("ConsciousAgent: No conscious-info memories found")
                return False

            # Copy each conscious-info memory directly to short-term memory
            copied_count = 0
            for memory_row in conscious_memories:
                if self.use_heuristics and not self._passes_lightweight_heuristics(
                    db_manager, namespace, memory_row
                ):
                    logger.debug(
                        f"ConsciousAgent: Skipping memory {memory_row[0]} - failed lightweight heuristics"
                    )
                    continue

                success = await self._copy_memory_to_short_term(
                    db_manager,
                    namespace,
                    memory_row,
                    prevalidated=self.use_heuristics,
                )
                if success:
                    copied_count += 1

            # Mark memories as processed
            memory_ids = [
                row[0] for row in conscious_memories
            ]  # memory_id is first column
            await self._mark_memories_processed(db_manager, memory_ids, namespace)

            self.context_initialized = True
            logger.info(
                f"ConsciousAgent: Copied {copied_count} conscious-info memories to short-term memory"
            )

            return copied_count > 0

        except Exception as e:
            logger.error(f"ConsciousAgent: Conscious ingest failed: {e}")
            return False

    async def initialize_existing_conscious_memories(
        self, db_manager, namespace: str = "default"
    ) -> bool:
        """
        Initialize by copying ALL existing conscious-info memories to short-term memory
        This is called when both auto_ingest=True and conscious_ingest=True
        to ensure essential conscious information is immediately available

        Args:
            db_manager: Database manager instance
            namespace: Memory namespace

        Returns:
            True if memories were processed, False otherwise
        """
        if not getattr(db_manager, "enable_short_term", True):
            logger.info("Short-term memory disabled; skipping initialization")
            return False
        try:
            from sqlalchemy import text

            with db_manager.get_connection() as connection:
                # Get ALL conscious-info labeled memories from long-term memory
                cursor = connection.execute(
                    text(
                        """SELECT memory_id, processed_data, summary, searchable_content,
                              importance_score, created_at, x_coord, y_coord, z_coord,
                              symbolic_anchors
                       FROM long_term_memory
                       WHERE namespace = :namespace AND classification = 'conscious-info'
                       ORDER BY importance_score DESC, created_at DESC"""
                    ),
                    {"namespace": namespace},
                )
                existing_conscious_memories = cursor.fetchall()

            if not existing_conscious_memories:
                logger.debug(
                    "ConsciousAgent: No existing conscious-info memories found for initialization"
                )
                return False

            copied_count = 0
            for memory_row in existing_conscious_memories:
                if self.use_heuristics and not self._passes_lightweight_heuristics(
                    db_manager, namespace, memory_row
                ):
                    logger.debug(
                        f"ConsciousAgent: Skipping existing memory {memory_row[0]} - failed lightweight heuristics"
                    )
                    continue

                success = await self._copy_memory_to_short_term(
                    db_manager,
                    namespace,
                    memory_row,
                    prevalidated=self.use_heuristics,
                )
                if success:
                    copied_count += 1

            if copied_count > 0:
                logger.info(
                    f"ConsciousAgent: Initialized {copied_count} existing conscious-info memories to short-term memory"
                )
                return True
            else:
                logger.debug(
                    "ConsciousAgent: No new conscious memories to initialize (all were duplicates)"
                )
                return False

        except Exception as e:
            logger.error(
                f"ConsciousAgent: Failed to initialize existing conscious memories: {e}"
            )
            return False

    async def check_for_context_updates(
        self, db_manager, namespace: str = "default"
    ) -> bool:
        """
        Check for new conscious-info memories and copy them to short-term memory

        Args:
            db_manager: Database manager instance
            namespace: Memory namespace

        Returns:
            True if new memories were copied, False otherwise
        """
        if not getattr(db_manager, "enable_short_term", True):
            return False
        try:
            # Get unprocessed conscious memories
            new_memories = await self._get_unprocessed_conscious_memories(
                db_manager, namespace
            )

            if not new_memories:
                return False

            # Copy each new memory directly to short-term memory
            copied_count = 0
            for memory_row in new_memories:
                if self.use_heuristics and not self._passes_lightweight_heuristics(
                    db_manager, namespace, memory_row
                ):
                    logger.debug(
                        f"ConsciousAgent: Skipping new memory {memory_row[0]} - failed lightweight heuristics"
                    )
                    continue

                success = await self._copy_memory_to_short_term(
                    db_manager,
                    namespace,
                    memory_row,
                    prevalidated=self.use_heuristics,
                )
                if success:
                    copied_count += 1

            # Mark new memories as processed
            memory_ids = [row[0] for row in new_memories]  # memory_id is first column
            await self._mark_memories_processed(db_manager, memory_ids, namespace)

            logger.info(
                f"ConsciousAgent: Copied {copied_count} new conscious-info memories to short-term memory"
            )
            return copied_count > 0

        except Exception as e:
            logger.error(f"ConsciousAgent: Context update failed: {e}")
            return False

    async def _get_conscious_memories(self, db_manager, namespace: str) -> list[tuple]:
        """Get all conscious-info labeled memories from long-term memory"""
        try:
            from sqlalchemy import text

            with db_manager.get_connection() as connection:
                cursor = connection.execute(
                    text(
                        """SELECT memory_id, processed_data, summary, searchable_content,
                              importance_score, created_at, x_coord, y_coord, z_coord,
                              symbolic_anchors
                       FROM long_term_memory
                       WHERE namespace = :namespace AND classification = 'conscious-info'
                       ORDER BY importance_score DESC, created_at DESC"""
                    ),
                    {"namespace": namespace},
                )
                return cursor.fetchall()

        except Exception as e:
            logger.error(f"ConsciousAgent: Failed to get conscious memories: {e}")
            return []

    async def _get_unprocessed_conscious_memories(
        self, db_manager, namespace: str
    ) -> list[tuple]:
        """Get unprocessed conscious-info labeled memories from long-term memory"""
        try:
            from sqlalchemy import text

            with db_manager.get_connection() as connection:
                cursor = connection.execute(
                    text(
                        """SELECT memory_id, processed_data, summary, searchable_content,
                              importance_score, created_at, x_coord, y_coord, z_coord,
                              symbolic_anchors
                       FROM long_term_memory
                       WHERE namespace = :namespace AND classification = 'conscious-info'
                       AND conscious_processed = :conscious_processed
                       ORDER BY importance_score DESC, created_at DESC"""
                    ),
                    {"namespace": namespace, "conscious_processed": False},
                )
                return cursor.fetchall()

        except Exception as e:
            logger.error(f"ConsciousAgent: Failed to get unprocessed memories: {e}")
            return []

    async def _copy_memory_to_short_term(
        self,
        db_manager,
        namespace: str,
        memory_row: tuple,
        *,
        prevalidated: bool = False,
    ) -> bool:
        """Copy a conscious memory directly to short-term memory with duplicate filtering"""
        if not getattr(db_manager, "enable_short_term", True):
            return False
        try:
            if self.use_heuristics and not prevalidated:
                if not self._passes_lightweight_heuristics(
                    db_manager, namespace, memory_row
                ):
                    logger.debug(
                        f"ConsciousAgent: Skipping memory {memory_row[0]} during copy - failed lightweight heuristics"
                    )
                    return False

            (
                memory_id,
                processed_data,
                summary,
                searchable_content,
                importance_score,
                _,
                x_coord,
                y_coord,
                z_coord,
                symbolic_anchors,
            ) = memory_row

            anchors_value = symbolic_anchors
            if isinstance(anchors_value, str):
                try:
                    anchors_value = json.loads(anchors_value)
                except json.JSONDecodeError:
                    anchors_value = [
                        anchor.strip()
                        for anchor in anchors_value.split(",")
                        if anchor.strip()
                    ]

            if anchors_value is None:
                anchors_value = []

            from sqlalchemy import text

            with db_manager.get_connection() as connection:
                # Check if similar content already exists in short-term memory
                existing_check = connection.execute(
                    text(
                        """SELECT COUNT(*) FROM short_term_memory
                           WHERE namespace = :namespace
                           AND category_primary = :category_primary
                           AND (searchable_content = :searchable_content
                                OR summary = :summary)"""
                    ),
                    {
                        "namespace": namespace,
                        "category_primary": CONSCIOUS_CONTEXT_CATEGORY,
                        "searchable_content": searchable_content,
                        "summary": summary,
                    },
                )

                existing_count = existing_check.scalar()
                if existing_count > 0:
                    logger.debug(
                        f"ConsciousAgent: Skipping duplicate memory {memory_id} - similar content already exists in short-term memory"
                    )
                    return False

                # Create short-term memory ID
                short_term_id = (
                    f"conscious_{memory_id}_{int(datetime.now().timestamp())}"
                )

                # Insert directly into short-term memory with essential conscious category
                connection.execute(
                    text(
                        """INSERT INTO short_term_memory (
                        memory_id, processed_data, importance_score, category_primary,
                        retention_type, namespace, created_at, expires_at,
                        searchable_content, summary, is_permanent_context,
                        x_coord, y_coord, z_coord, symbolic_anchors
                    ) VALUES (:memory_id, :processed_data, :importance_score, :category_primary,
                        :retention_type, :namespace, :created_at, :expires_at,
                        :searchable_content, :summary, :is_permanent_context,
                        :x_coord, :y_coord, :z_coord, :symbolic_anchors)"""
                    ),
                    {
                        "memory_id": short_term_id,
                        "processed_data": (
                            json.dumps(processed_data)
                            if isinstance(processed_data, dict)
                            else processed_data
                        ),
                        "importance_score": importance_score,
                        "category_primary": CONSCIOUS_CONTEXT_CATEGORY,
                        "retention_type": "permanent",
                        "namespace": namespace,
                        "created_at": datetime.now().isoformat(),
                        "expires_at": None,  # No expiration (permanent)
                        "searchable_content": searchable_content,  # Copy exact searchable_content
                        "summary": summary,  # Copy exact summary
                        "is_permanent_context": True,  # is_permanent_context = True
                        "x_coord": x_coord,
                        "y_coord": y_coord,
                        "z_coord": z_coord,
                        "symbolic_anchors": json.dumps(anchors_value),
                    },
                )
                connection.commit()

            logger.debug(
                f"ConsciousAgent: Copied memory {memory_id} to short-term as {short_term_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"ConsciousAgent: Failed to copy memory {memory_row[0]} to short-term: {e}"
            )
            return False

    def _parse_created_at(self, created_at_value) -> datetime | None:
        """Best-effort parsing of created_at values."""
        if isinstance(created_at_value, datetime):
            return created_at_value

        if isinstance(created_at_value, str):
            iso_value = created_at_value.strip()
            if not iso_value:
                return None
            try:
                return datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
            except ValueError:
                for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                    try:
                        return datetime.strptime(iso_value, fmt)
                    except ValueError:
                        continue
        return None

    def _passes_lightweight_heuristics(
        self, db_manager, namespace: str, memory_row: tuple
    ) -> bool:
        """Apply lightweight heuristics before copying long-term memories."""
        candidate_rows = []

        try:
            memory_id = memory_row[0]
            summary = memory_row[2]
            searchable_content = memory_row[3]
            created_at = memory_row[5]
        except (TypeError, IndexError):
            logger.debug("ConsciousAgent: Invalid memory row received for heuristics")
            return False

        summary_text = (summary or "").strip()
        content_text = (searchable_content or "").strip()

        if len(summary_text) < self.heuristic_config["min_summary_chars"]:
            logger.debug(
                f"ConsciousAgent: Memory {memory_id} skipped - summary below minimum length"
            )
            return False

        if len(content_text) < self.heuristic_config["min_searchable_chars"]:
            logger.debug(
                f"ConsciousAgent: Memory {memory_id} skipped - searchable content below minimum length"
            )
            return False

        created_at_dt = self._parse_created_at(created_at)
        if not created_at_dt:
            logger.debug(
                f"ConsciousAgent: Memory {memory_id} skipped - created_at could not be parsed"
            )
            return False

        if created_at_dt.tzinfo is None:
            created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)
        else:
            created_at_dt = created_at_dt.astimezone(timezone.utc)

        now_utc = datetime.now(timezone.utc)
        max_age = timedelta(minutes=self.heuristic_config["max_age_minutes"])
        if now_utc - created_at_dt > max_age:
            logger.debug(
                f"ConsciousAgent: Memory {memory_id} skipped - created_at outside recency window"
            )
            return False

        try:
            from sqlalchemy import text

            with db_manager.get_connection() as connection:
                params = {
                    "namespace": namespace,
                    "memory_id": memory_id,
                    "exact_content": content_text,
                }

                exact_match = connection.execute(
                    text(
                        """SELECT memory_id FROM long_term_memory
                        WHERE namespace = :namespace
                          AND memory_id != :memory_id
                          AND searchable_content = :exact_content
                        LIMIT 1"""
                    ),
                    params,
                ).fetchone()

                if exact_match:
                    return True

                snippet = content_text[
                    : self.heuristic_config["match_snippet_chars"]
                ].strip()
                if len(snippet) >= self.heuristic_config["min_match_chars"]:
                    like_match = connection.execute(
                        text(
                            """SELECT memory_id FROM long_term_memory
                            WHERE namespace = :namespace
                              AND memory_id != :memory_id
                              AND searchable_content LIKE :content_like
                            LIMIT 1"""
                        ),
                        {
                            "namespace": namespace,
                            "memory_id": memory_id,
                            "content_like": f"%{snippet}%",
                        },
                    ).fetchone()
                    if like_match:
                        return True

                candidate_rows = connection.execute(
                    text(
                        """SELECT memory_id, searchable_content FROM long_term_memory
                        WHERE namespace = :namespace
                          AND memory_id != :memory_id
                        ORDER BY created_at DESC
                        LIMIT :limit"""
                    ),
                    {
                        "namespace": namespace,
                        "memory_id": memory_id,
                        "limit": self.heuristic_config["candidate_scan_limit"],
                    },
                ).fetchall()

        except Exception as exc:
            logger.warning(
                f"ConsciousAgent: Lightweight heuristic content lookup failed for {memory_id}: {exc}"
            )
            return False

        for _other_id, other_content in candidate_rows:
            other_text = (other_content or "").strip()
            if not other_text:
                continue
            if other_text in content_text or content_text in other_text:
                return True
            if summary_text and summary_text in other_text:
                return True

        logger.debug(
            f"ConsciousAgent: Memory {memory_id} skipped - no supporting long-term content match found"
        )
        return False

    async def _mark_memories_processed(
        self, db_manager, memory_ids: list[str], namespace: str
    ):
        """Mark memories as processed for conscious context"""
        try:
            from sqlalchemy import text

            with db_manager.get_connection() as connection:
                for memory_id in memory_ids:
                    connection.execute(
                        text(
                            """UPDATE long_term_memory
                           SET conscious_processed = :conscious_processed
                           WHERE memory_id = :memory_id AND namespace = :namespace"""
                        ),
                        {
                            "memory_id": memory_id,
                            "namespace": namespace,
                            "conscious_processed": True,
                        },
                    )
                connection.commit()

        except Exception as e:
            logger.error(f"ConsciousAgent: Failed to mark memories processed: {e}")
