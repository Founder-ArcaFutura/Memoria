"""Conscious memory management utilities."""

from __future__ import annotations

import asyncio
import json
import threading

from loguru import logger

from .constants import CONSCIOUS_CONTEXT_CATEGORY


class ConsciousManager:
    """Handle conscious memory initialization and background analysis."""

    def __init__(self, memoria: "Memoria"):
        self.memoria = memoria
        self._background_task = None
        self._conscious_init_pending = False
        self._initialized = False
        self._analysis_interval = getattr(
            memoria, "conscious_analysis_interval_seconds", 6 * 60 * 60
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Initialize conscious memory and start background analysis."""
        if not self.memoria.conscious_ingest or not self.memoria.conscious_agent:
            return

        if not self._initialized:
            self._initialize_conscious_memory()

        if self.memoria._enabled:
            # If initialization is still running attach a callback, otherwise start now
            if self._background_task and not self._background_task.done():
                self._background_task.add_done_callback(
                    lambda _: self._start_background_analysis()
                )
            else:
                self._start_background_analysis()

    def stop(self) -> None:
        """Stop any background analysis."""
        self._stop_background_analysis()

    def trigger_analysis(self):
        """Manually trigger conscious context ingestion."""
        if not self.memoria.conscious_ingest or not self.memoria.conscious_agent:
            logger.warning("Conscious ingestion not enabled or agent not available")
            return

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self.memoria.conscious_agent.run_conscious_ingest(
                    self.memoria.db_manager, self.memoria.namespace
                )
            )
            logger.info("Conscious context ingestion triggered")
            return task
        except RuntimeError:
            # No running loop - execute in a separate thread
            def run_analysis():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(
                        self.memoria.conscious_agent.run_conscious_ingest(
                            self.memoria.db_manager, self.memoria.namespace
                        )
                    )
                finally:
                    new_loop.close()

            thread = threading.Thread(target=run_analysis)
            thread.start()
            logger.info("Conscious context ingestion triggered in separate thread")

    def check_deferred_initialization(self) -> None:
        """Check and handle deferred conscious memory initialization."""
        if self._conscious_init_pending and self.memoria.conscious_agent:
            try:
                loop = asyncio.get_running_loop()
                if self._background_task is None or self._background_task.done():
                    self._background_task = loop.create_task(
                        self._run_conscious_initialization()
                    )
                    logger.debug(
                        "Conscious-ingest: Deferred initialization task started"
                    )
                    self._conscious_init_pending = False
            except RuntimeError:
                logger.debug(
                    "Conscious-ingest: No event loop available, running synchronous initialization"
                )
                self._run_synchronous_conscious_initialization()
                self._conscious_init_pending = False

    def is_running(self) -> bool:
        """Return True if background analysis is active."""
        return bool(self._background_task and not self._background_task.done())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize_conscious_memory(self) -> None:
        """Initialize conscious memory by running conscious agent analysis."""
        try:
            logger.info(
                "Conscious-ingest: Starting conscious agent analysis at startup"
            )
            try:
                loop = asyncio.get_running_loop()
                if self._background_task is None or self._background_task.done():
                    self._background_task = loop.create_task(
                        self._run_conscious_initialization()
                    )
                    logger.debug(
                        "Conscious-ingest: Background initialization task started"
                    )
            except RuntimeError:
                logger.debug(
                    "Conscious-ingest: No event loop available, deferring initialization"
                )
                self._conscious_init_pending = True
        except Exception as e:
            logger.error(f"Failed to initialize conscious memory: {e}")

    async def _run_conscious_initialization(self) -> None:
        """Run conscious agent initialization in background."""
        try:
            if not self.memoria.conscious_agent:
                return

            if self.memoria.auto_ingest and self.memoria.conscious_ingest:
                logger.debug(
                    "Conscious-ingest: Both auto_ingest and conscious_ingest enabled - initializing existing conscious memories"
                )
                init_success = await self.memoria.conscious_agent.initialize_existing_conscious_memories(
                    self.memoria.db_manager, self.memoria.namespace
                )
                if init_success:
                    logger.info(
                        "Conscious-ingest: Existing conscious-info memories initialized to short-term memory"
                    )

            logger.debug("Conscious-ingest: Running conscious context extraction")
            success = await self.memoria.conscious_agent.run_conscious_ingest(
                self.memoria.db_manager, self.memoria.namespace
            )

            if success:
                logger.info(
                    "Conscious-ingest: Conscious memories copied to short-term memory"
                )
            else:
                logger.info("Conscious-ingest: No conscious context found")

            self._initialized = True
        except Exception as e:
            logger.error(f"Conscious agent initialization failed: {e}")

    def _run_synchronous_conscious_initialization(self) -> None:
        """Run conscious agent initialization synchronously."""
        try:
            if not self.memoria.conscious_agent:
                return

            if self.memoria.auto_ingest and self.memoria.conscious_ingest:
                logger.info(
                    "Conscious-ingest: Both auto_ingest and conscious_ingest enabled - initializing existing conscious memories"
                )
                self._initialize_existing_conscious_memories_sync()

            logger.debug(
                "Conscious-ingest: Synchronous conscious context extraction completed"
            )
            self._initialized = True
        except Exception as e:
            logger.error(f"Synchronous conscious agent initialization failed: {e}")

    def _initialize_existing_conscious_memories_sync(self) -> bool:
        """Synchronously initialize existing conscious-info memories."""
        try:
            from sqlalchemy import text

            with self.memoria.db_manager.get_connection() as connection:
                cursor = connection.execute(
                    text(
                        """SELECT memory_id, processed_data, summary, searchable_content,
                              importance_score, created_at, x_coord, y_coord, z_coord,
                              symbolic_anchors
                       FROM long_term_memory
                       WHERE namespace = :namespace AND classification = 'conscious-info'
                       ORDER BY importance_score DESC, created_at DESC"""
                    ),
                    {"namespace": self.memoria.namespace or "default"},
                )
                existing_conscious_memories = cursor.fetchall()

            if not existing_conscious_memories:
                logger.debug(
                    "Conscious-ingest: No existing conscious-info memories found for initialization"
                )
                return False

            copied_count = 0
            for memory_row in existing_conscious_memories:
                if self._copy_memory_to_short_term_sync(memory_row):
                    copied_count += 1

            if copied_count > 0:
                logger.info(
                    f"Conscious-ingest: Initialized {copied_count} existing conscious-info memories to short-term memory"
                )
                return True
            logger.debug(
                "Conscious-ingest: No new conscious memories to initialize (all were duplicates)"
            )
            return False
        except Exception as e:
            logger.error(
                f"Conscious-ingest: Failed to initialize existing conscious memories: {e}"
            )
            return False

    def _copy_memory_to_short_term_sync(self, memory_row: tuple) -> bool:
        """Synchronously copy a conscious memory to short-term memory."""
        try:
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

            from datetime import datetime

            from sqlalchemy import text

            with self.memoria.db_manager.get_connection() as connection:
                existing_check = connection.execute(
                    text(
                        """SELECT COUNT(*) FROM short_term_memory
                           WHERE namespace = :namespace
                           AND category_primary = :category_primary
                           AND (searchable_content = :searchable_content
                                OR summary = :summary)"""
                    ),
                    {
                        "namespace": self.memoria.namespace or "default",
                        "category_primary": CONSCIOUS_CONTEXT_CATEGORY,
                        "searchable_content": searchable_content,
                        "summary": summary,
                    },
                )

                if existing_check.scalar() > 0:
                    logger.debug(
                        f"Conscious-ingest: Skipping duplicate memory {memory_id} - similar content already exists in short-term memory"
                    )
                    return False

                short_term_id = (
                    f"conscious_{memory_id}_{int(datetime.now().timestamp())}"
                )

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
                        "processed_data": processed_data,
                        "importance_score": importance_score,
                        "category_primary": CONSCIOUS_CONTEXT_CATEGORY,
                        "retention_type": "permanent",
                        "namespace": self.memoria.namespace or "default",
                        "created_at": datetime.now().isoformat(),
                        "expires_at": None,
                        "searchable_content": searchable_content,
                        "summary": summary,
                        "is_permanent_context": True,
                        "x_coord": x_coord,
                        "y_coord": y_coord,
                        "z_coord": z_coord,
                        "symbolic_anchors": json.dumps(anchors_value),
                    },
                )
                connection.commit()

            logger.debug(
                f"Conscious-ingest: Copied memory {memory_id} to short-term as {short_term_id}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Conscious-ingest: Failed to copy memory {memory_row[0]} to short-term: {e}"
            )
            return False

    # ------------------------------------------------------------------
    # Background analysis
    # ------------------------------------------------------------------
    def _start_background_analysis(self) -> None:
        """Start the background conscious agent analysis task."""
        try:
            if self._background_task and not self._background_task.done():
                logger.debug("Background analysis task already running")
                return

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:

                def run_background_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        new_loop.run_until_complete(self._background_analysis_loop())
                    except Exception as e:
                        logger.error(f"Background analysis loop failed: {e}")
                    finally:
                        new_loop.close()

                thread = threading.Thread(target=run_background_loop, daemon=True)
                thread.start()
                logger.info("Background analysis started in separate thread")
                return

            self._background_task = loop.create_task(self._background_analysis_loop())
            self._background_task.add_done_callback(
                self._handle_background_task_completion
            )
            logger.info("Background analysis task started")
        except Exception as e:
            logger.error(f"Failed to start background analysis: {e}")

    def _handle_background_task_completion(self, task) -> None:
        """Handle background task completion and cleanup."""
        try:
            if task.exception():
                logger.error(f"Background task failed: {task.exception()}")
        except asyncio.CancelledError:
            logger.debug("Background task was cancelled")
        except Exception as e:
            logger.error(f"Error handling background task completion: {e}")

    def _stop_background_analysis(self) -> None:
        """Stop the background analysis task."""
        try:
            if self._background_task and not self._background_task.done():
                self._background_task.cancel()
                logger.info("Background analysis task stopped")
        except Exception as e:
            logger.error(f"Failed to stop background analysis: {e}")

    async def _background_analysis_loop(self) -> None:
        """Background analysis loop for memory processing."""
        try:
            logger.debug("Background analysis loop started")

            if self.memoria.conscious_ingest and self.memoria.conscious_agent:
                while True:
                    try:
                        await asyncio.sleep(self._analysis_interval)
                        await self.memoria.conscious_agent.run_conscious_ingest(
                            self.memoria.db_manager, self.memoria.namespace
                        )
                        logger.debug("Periodic conscious analysis completed")
                    except asyncio.CancelledError:
                        logger.debug("Background analysis loop cancelled")
                        break
                    except Exception as e:
                        logger.error(f"Background analysis error: {e}")
                        await asyncio.sleep(60)
            else:
                while True:
                    await asyncio.sleep(3600)
        except asyncio.CancelledError:
            logger.debug("Background analysis loop cancelled")
        except Exception as e:
            logger.error(f"Background analysis loop failed: {e}")


__all__ = ["ConsciousManager"]
