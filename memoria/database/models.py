"""
SQLAlchemy models for the Memoria 0.9 alpha (0.9.0a0) release.
Provides cross-database compatibility using SQLAlchemy ORM.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, synonym

from memoria.schemas.constants import X_AXIS, Y_AXIS, Z_AXIS

Base: Any = declarative_base()


class Team(Base):
    """Team metadata used to scope conversations and memories."""

    __tablename__ = "teams"

    team_id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    chat_histories = relationship("ChatHistory", back_populates="team")
    short_term_memories = relationship("ShortTermMemory", back_populates="team")
    long_term_memories = relationship("LongTermMemory", back_populates="team")
    workspace = relationship("Workspace", back_populates="team", uselist=False)
    policy_artifacts = relationship("PolicyArtifact", back_populates="team")

    __table_args__ = (
        Index("idx_team_slug", "slug", unique=True),
        Index("idx_team_created", "created_at"),
    )


class Workspace(Base):
    """Workspace metadata providing ownership and scope for data isolation."""

    __tablename__ = "workspaces"

    workspace_id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(String(255))
    team_id = Column(String(255), ForeignKey("teams.team_id", ondelete="SET NULL"))
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    team = relationship("Team", back_populates="workspace")
    members = relationship(
        "WorkspaceMember", back_populates="workspace", cascade="all, delete-orphan"
    )
    chat_histories = relationship("ChatHistory", back_populates="workspace")
    short_term_memories = relationship("ShortTermMemory", back_populates="workspace")
    long_term_memories = relationship("LongTermMemory", back_populates="workspace")
    spatial_metadata_entries = relationship(
        "SpatialMetadata", back_populates="workspace"
    )
    access_events = relationship("MemoryAccessEvent", back_populates="workspace")
    retention_audits = relationship("RetentionPolicyAudit", back_populates="workspace")
    policy_artifacts = relationship("PolicyArtifact", back_populates="workspace")

    __table_args__ = (
        Index("idx_workspace_slug", "slug", unique=True),
        Index("idx_workspace_owner", "owner_id"),
        Index("idx_workspace_team", "team_id"),
        Index("idx_workspace_created", "created_at"),
    )


class WorkspaceMember(Base):
    """Members of a workspace with role metadata."""

    __tablename__ = "workspace_members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(
        String(255),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="member")
    is_admin = Column(Boolean, nullable=False, default=False)
    joined_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    workspace = relationship("Workspace", back_populates="members")

    __table_args__ = (
        Index("idx_workspace_member_workspace", "workspace_id"),
        Index("idx_workspace_member_user", "user_id"),
        UniqueConstraint("workspace_id", "user_id", name="uq_workspace_member"),
    )


class ChatHistory(Base):
    """Chat history table - stores all conversations"""

    __tablename__ = "chat_history"

    chat_id = Column(String(255), primary_key=True)
    user_input = Column(Text, nullable=False)
    ai_output = Column(Text, nullable=False)
    model = Column(String(255), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    session_id = Column(String(255), nullable=False)
    namespace = Column(String(255), nullable=False, default="default")
    tokens_used = Column(Integer, default=0)
    metadata_ = Column("metadata", JSON)
    __mapper_args__ = {"properties": {"metadata": metadata_}}
    metadata_json = synonym("metadata")
    team_id = Column(String(255), ForeignKey("teams.team_id", ondelete="SET NULL"))
    workspace_id = Column(
        String(255), ForeignKey("workspaces.workspace_id", ondelete="SET NULL")
    )

    # Relationships
    short_term_memories = relationship(
        "ShortTermMemory", back_populates="chat", cascade="all, delete-orphan"
    )
    team = relationship("Team", back_populates="chat_histories")
    workspace = relationship("Workspace", back_populates="chat_histories")

    # Indexes
    __table_args__ = (
        Index("idx_chat_namespace_session", "namespace", "session_id"),
        Index("idx_chat_team_namespace", "team_id", "namespace"),
        Index("idx_chat_timestamp", "timestamp"),
        Index("idx_chat_model", "model"),
        Index("idx_chat_team", "team_id"),
        Index("idx_chat_workspace", "workspace_id"),
        Index("idx_chat_workspace_namespace", "workspace_id", "namespace"),
    )


class ShortTermMemory(Base):
    """Short-term memory table with expiration"""

    __tablename__ = "short_term_memory"

    memory_id = Column(String(255), primary_key=True)
    chat_id = Column(
        String(255), ForeignKey("chat_history.chat_id", ondelete="SET NULL")
    )
    processed_data = Column(JSON, nullable=False)
    importance_score = Column(Float, nullable=False, default=0.5)
    category_primary = Column(String(255), nullable=False)
    retention_type = Column(String(50), nullable=False, default="short_term")
    namespace = Column(String(255), nullable=False, default="default")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime)
    searchable_content = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    is_permanent_context = Column(Boolean, default=False)
    x_coord = Column(Float, doc=X_AXIS.description)
    y_coord = Column(Float, doc=Y_AXIS.description)
    z_coord = Column(Float, doc=Z_AXIS.description)
    symbolic_anchors = Column(JSON)
    embedding = Column(JSON)
    team_id = Column(String(255), ForeignKey("teams.team_id", ondelete="SET NULL"))
    workspace_id = Column(
        String(255), ForeignKey("workspaces.workspace_id", ondelete="SET NULL")
    )

    # Relationships
    chat = relationship("ChatHistory", back_populates="short_term_memories")
    team = relationship("Team", back_populates="short_term_memories")
    workspace = relationship("Workspace", back_populates="short_term_memories")

    # Indexes
    __table_args__ = (
        Index("idx_short_term_namespace", "namespace"),
        Index("idx_short_term_team_namespace", "team_id", "namespace"),
        Index("idx_short_term_category", "category_primary"),
        Index("idx_short_term_importance", "importance_score"),
        Index("idx_short_term_expires", "expires_at"),
        Index("idx_short_term_created", "created_at"),
        Index("idx_short_term_access", "access_count", "last_accessed"),
        Index("idx_short_term_permanent", "is_permanent_context"),
        Index(
            "idx_short_term_namespace_category",
            "namespace",
            "category_primary",
            "importance_score",
        ),
        Index("idx_short_term_coords", "x_coord", "y_coord", "z_coord"),
        Index("idx_short_term_team", "team_id"),
        Index("idx_short_term_namespace_team", "namespace", "team_id"),
        Index("idx_short_term_workspace", "workspace_id"),
        Index("idx_short_term_namespace_workspace", "namespace", "workspace_id"),
        CheckConstraint(
            f"(y_coord IS NULL OR (y_coord >= {Y_AXIS.min} AND y_coord <= {Y_AXIS.max}))",
            name="ck_short_term_y_coord_range",
        ),
        CheckConstraint(
            f"(z_coord IS NULL OR (z_coord >= {Z_AXIS.min} AND z_coord <= {Z_AXIS.max}))",
            name="ck_short_term_z_coord_range",
        ),
    )


class LongTermMemory(Base):
    """Long-term memory table with enhanced classification"""

    __tablename__ = "long_term_memory"

    memory_id = Column(String(255), primary_key=True)
    original_chat_id = Column(String(255))
    processed_data = Column(JSON, nullable=False)
    importance_score = Column(Float, nullable=False, default=0.5)
    category_primary = Column(String(255), nullable=False)
    retention_type = Column(String(50), nullable=False, default="long_term")
    namespace = Column(String(255), nullable=False, default="default")
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime)
    searchable_content = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    novelty_score = Column(Float, default=0.5)
    relevance_score = Column(Float, default=0.5)
    actionability_score = Column(Float, default=0.5)
    x_coord = Column(Float, doc=X_AXIS.description)
    y_coord = Column(Float, doc=Y_AXIS.description)
    z_coord = Column(Float, doc=Z_AXIS.description)
    symbolic_anchors = Column(JSON)
    embedding = Column(JSON)
    documents_json = Column(JSON)
    team_id = Column(String(255), ForeignKey("teams.team_id", ondelete="SET NULL"))
    workspace_id = Column(
        String(255), ForeignKey("workspaces.workspace_id", ondelete="SET NULL")
    )

    # Enhanced Classification Fields
    classification = Column(String(50), nullable=False, default="conversational")
    memory_importance = Column(String(20), nullable=False, default="medium")
    topic = Column(String(255))
    entities_json = Column(JSON)
    keywords_json = Column(JSON)

    # Conscious Context Flags
    is_user_context = Column(Boolean, default=False)
    is_preference = Column(Boolean, default=False)
    is_skill_knowledge = Column(Boolean, default=False)
    is_current_project = Column(Boolean, default=False)
    promotion_eligible = Column(Boolean, default=False)

    # Memory Management
    duplicate_of = Column(String(255))
    supersedes_json = Column(JSON)
    related_memories_json = Column(JSON)

    # Technical Metadata
    confidence_score = Column(Float, default=0.8)
    extraction_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    classification_reason = Column(Text)

    # Processing Status
    processed_for_duplicates = Column(Boolean, default=False)
    conscious_processed = Column(Boolean, default=False)

    # Relationships
    team = relationship("Team", back_populates="long_term_memories")
    workspace = relationship("Workspace", back_populates="long_term_memories")

    # Indexes
    __table_args__ = (
        Index("idx_long_term_namespace", "namespace"),
        Index("idx_long_term_team_namespace", "team_id", "namespace"),
        Index("idx_long_term_category", "category_primary"),
        Index("idx_long_term_importance", "importance_score"),
        Index("idx_long_term_created", "created_at"),
        Index("idx_long_term_timestamp", "timestamp"),
        Index("idx_long_term_access", "access_count", "last_accessed"),
        Index(
            "idx_long_term_scores",
            "novelty_score",
            "relevance_score",
            "actionability_score",
        ),
        Index("idx_long_term_classification", "classification"),
        Index("idx_long_term_memory_importance", "memory_importance"),
        Index("idx_long_term_topic", "topic"),
        Index(
            "idx_long_term_conscious_flags",
            "is_user_context",
            "is_preference",
            "is_skill_knowledge",
            "promotion_eligible",
        ),
        Index("idx_long_term_conscious_processed", "conscious_processed"),
        Index("idx_long_term_duplicates", "processed_for_duplicates"),
        Index("idx_long_term_confidence", "confidence_score"),
        Index(
            "idx_long_term_namespace_category",
            "namespace",
            "category_primary",
            "importance_score",
        ),
        Index("idx_long_term_coords", "x_coord", "y_coord", "z_coord"),
        Index("idx_long_term_team", "team_id"),
        Index("idx_long_term_namespace_team", "namespace", "team_id"),
        Index("idx_long_term_workspace", "workspace_id"),
        Index("idx_long_term_namespace_workspace", "namespace", "workspace_id"),
        CheckConstraint(
            f"(y_coord IS NULL OR (y_coord >= {Y_AXIS.min} AND y_coord <= {Y_AXIS.max}))",
            name="ck_long_term_y_coord_range",
        ),
        CheckConstraint(
            f"(z_coord IS NULL OR (z_coord >= {Z_AXIS.min} AND z_coord <= {Z_AXIS.max}))",
            name="ck_long_term_z_coord_range",
        ),
    )


class SpatialMetadata(Base):
    """Auxiliary table storing spatial coordinates and symbolic anchors."""

    __tablename__ = "spatial_metadata"

    memory_id = Column(String(255), primary_key=True)
    namespace = Column(String(255), nullable=False, default="default")
    team_id = Column(String(255), ForeignKey("teams.team_id", ondelete="SET NULL"))
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    x = Column(Float, doc=X_AXIS.description)
    y = Column(Float, doc=Y_AXIS.description)
    z = Column(Float, doc=Z_AXIS.description)
    symbolic_anchors = Column(JSON)

    workspace_id = Column(
        String(255), ForeignKey("workspaces.workspace_id", ondelete="SET NULL")
    )

    workspace = relationship("Workspace", back_populates="spatial_metadata_entries")

    __table_args__ = (
        Index("idx_spatial_namespace", "namespace"),
        Index("idx_spatial_team_namespace", "team_id", "namespace"),
        Index("idx_spatial_workspace_namespace", "workspace_id", "namespace"),
        CheckConstraint(
            "namespace <> ''",
            name="ck_spatial_metadata_namespace_nonempty",
        ),
        CheckConstraint(
            f"(y IS NULL OR (y >= {Y_AXIS.min} AND y <= {Y_AXIS.max}))",
            name="ck_spatial_metadata_y_range",
        ),
        CheckConstraint(
            f"(z IS NULL OR (z >= {Z_AXIS.min} AND z <= {Z_AXIS.max}))",
            name="ck_spatial_metadata_z_range",
        ),
    )


class MemoryAccessEvent(Base):
    """Audit log recording memory retrieval and reinforcement events."""

    __tablename__ = "memory_access_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(String(255), nullable=False)
    namespace = Column(String(255), nullable=False, default="default")
    team_id = Column(String(255))
    accessed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    access_type = Column(String(50), nullable=False, default="retrieval")
    source = Column(String(50))
    metadata_json = Column(JSON)

    workspace_id = Column(
        String(255), ForeignKey("workspaces.workspace_id", ondelete="SET NULL")
    )

    workspace = relationship("Workspace", back_populates="access_events")

    __table_args__ = (
        Index("idx_access_events_memory", "memory_id"),
        Index("idx_access_events_namespace", "namespace", "accessed_at"),
        Index(
            "idx_access_events_team_namespace", "team_id", "namespace", "accessed_at"
        ),
        Index(
            "idx_access_events_workspace_namespace",
            "workspace_id",
            "namespace",
            "accessed_at",
        ),
    )


class RetentionPolicyAudit(Base):
    """Audit log capturing retention policy enforcement decisions."""

    __tablename__ = "retention_policy_audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(String(255))
    namespace = Column(String(255), nullable=False, default="default")
    policy_name = Column(String(255), nullable=False)
    action = Column(String(50), nullable=False, default="block")
    escalate_to = Column(String(255))
    details = Column(JSON)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    team_id = Column(String(255), ForeignKey("teams.team_id", ondelete="SET NULL"))
    workspace_id = Column(
        String(255), ForeignKey("workspaces.workspace_id", ondelete="SET NULL")
    )

    workspace = relationship("Workspace", back_populates="retention_audits")

    __table_args__ = (
        Index("idx_retention_audits_namespace", "namespace", "created_at"),
        Index("idx_retention_audits_policy", "policy_name", "created_at"),
    )


class PolicyArtifact(Base):
    """Persistent representation of structured policy definitions."""

    __tablename__ = "policy_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    artifact_type = Column(String(50), nullable=False)
    namespace = Column(String(255), nullable=False, default="*")
    payload = Column(JSON, nullable=False)
    schema_version = Column(String(50), nullable=False, default="1.0")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    created_by = Column(String(255))
    workspace_id = Column(
        String(255), ForeignKey("workspaces.workspace_id", ondelete="SET NULL")
    )
    team_id = Column(String(255), ForeignKey("teams.team_id", ondelete="SET NULL"))

    workspace = relationship("Workspace", back_populates="policy_artifacts")
    team = relationship("Team", back_populates="policy_artifacts")

    __table_args__ = (
        UniqueConstraint(
            "name", "namespace", "workspace_id", name="uq_policy_artifacts_identity"
        ),
        Index("idx_policy_artifacts_type", "artifact_type"),
        Index("idx_policy_artifacts_namespace", "namespace"),
    )


# Explicit memory-to-memory links forming traversable threads
class LinkMemoryThread(Base):
    """Links between memories used to form explicit threads."""

    __tablename__ = "link_memory_threads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_memory_id = Column(String(255), nullable=False)
    target_memory_id = Column(String(255), nullable=False)
    relation = Column(String(50), nullable=False, default="related")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_threads_source", "source_memory_id"),
        Index("idx_threads_target", "target_memory_id"),
    )


class ThreadEvent(Base):
    """Metadata describing a conversational thread ingestion."""

    __tablename__ = "thread_events"

    thread_id = Column(String(255), primary_key=True)
    namespace = Column(String(255), nullable=False, default="default")
    symbolic_anchors = Column(JSON)
    ritual_name = Column(String(255))
    ritual_phase = Column(String(255))
    ritual_metadata = Column(JSON)
    centroid_x = Column(Float)
    centroid_y = Column(Float)
    centroid_z = Column(Float)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    __table_args__ = (Index("idx_thread_events_namespace", "namespace"),)


class ThreadMessageLink(Base):
    """Mapping between ingested thread messages and stored memories."""

    __tablename__ = "thread_message_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(255), nullable=False)
    memory_id = Column(String(255), nullable=False)
    namespace = Column(String(255), nullable=False, default="default")
    sequence_index = Column(Integer, nullable=False)
    role = Column(String(50))
    anchor = Column(String(255))
    timestamp = Column(DateTime)

    __table_args__ = (
        Index("idx_thread_message_thread", "thread_id"),
        Index("idx_thread_message_memory", "memory_id"),
    )


# ---------------------------------------------------------------------------
# Cluster tables
# ---------------------------------------------------------------------------


class Cluster(Base):
    """Represents a group of related memories."""

    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    summary = Column(Text, nullable=False)
    centroid = Column(JSON)
    y_centroid = Column(Float, doc=Y_AXIS.description)
    z_centroid = Column(Float, doc=Z_AXIS.description)
    polarity = Column(Float)
    subjectivity = Column(Float)
    size = Column(Integer)
    avg_importance = Column(Float)
    update_count = Column(Integer, default=0)
    last_updated = Column(DateTime)
    weight = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    total_chars = Column(Integer, default=0)

    members = relationship(
        "ClusterMember", back_populates="cluster", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_clusters_weight", "weight"),
        Index("idx_clusters_last_updated", "last_updated"),
        CheckConstraint(
            f"(y_centroid IS NULL OR (y_centroid >= {Y_AXIS.min} AND y_centroid <= {Y_AXIS.max}))",
            name="ck_clusters_y_centroid_range",
        ),
        CheckConstraint(
            f"(z_centroid IS NULL OR (z_centroid >= {Z_AXIS.min} AND z_centroid <= {Z_AXIS.max}))",
            name="ck_clusters_z_centroid_range",
        ),
    )


class ClusterMember(Base):
    """Individual memory belonging to a cluster."""

    __tablename__ = "cluster_members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(String(255), nullable=False)
    anchor = Column(String(255))
    summary = Column(Text, nullable=False)
    tokens = Column(Integer, default=0)
    chars = Column(Integer, default=0)
    cluster_id = Column(
        Integer, ForeignKey("clusters.id", ondelete="CASCADE"), nullable=False
    )

    cluster = relationship("Cluster", back_populates="members")

    __table_args__ = (Index("idx_cluster_members_cluster", "cluster_id"),)


# Database-specific configurations
def configure_mysql_fulltext(engine):
    """Configure MySQL FULLTEXT indexes"""
    if engine.dialect.name == "mysql":
        with engine.connect() as conn:
            try:
                # Create FULLTEXT indexes for MySQL
                conn.exec_driver_sql(
                    "ALTER TABLE short_term_memory ADD FULLTEXT INDEX ft_short_term_search (searchable_content, summary, symbolic_anchors)"
                )
                conn.exec_driver_sql(
                    "ALTER TABLE long_term_memory ADD FULLTEXT INDEX ft_long_term_search (searchable_content, summary, symbolic_anchors)"
                )
                conn.exec_driver_sql(
                    "ALTER TABLE long_term_memory ADD FULLTEXT INDEX ft_long_term_topic (topic)"
                )
                conn.commit()
            except Exception:
                # Indexes might already exist
                pass


def configure_postgresql_fts(engine):
    """Configure PostgreSQL full-text search"""
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            try:
                # Add tsvector columns for PostgreSQL
                conn.exec_driver_sql(
                    "ALTER TABLE short_term_memory ADD COLUMN IF NOT EXISTS search_vector tsvector"
                )
                conn.exec_driver_sql(
                    "ALTER TABLE long_term_memory ADD COLUMN IF NOT EXISTS search_vector tsvector"
                )

                # Create GIN indexes
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS idx_short_term_search_vector ON short_term_memory USING GIN(search_vector)"
                )
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS idx_long_term_search_vector ON long_term_memory USING GIN(search_vector)"
                )

                # Create triggers to maintain tsvector
                conn.exec_driver_sql(
                    """
                    CREATE OR REPLACE FUNCTION update_short_term_search_vector() RETURNS trigger AS $$
                    BEGIN
                        NEW.search_vector := to_tsvector('english', COALESCE(NEW.searchable_content, '') || ' ' || COALESCE(NEW.summary, '') || ' ' || COALESCE(NEW.symbolic_anchors, ''));
                        RETURN NEW;
                    END
                    $$ LANGUAGE plpgsql;
                """
                )

                conn.exec_driver_sql(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_trigger
                            WHERE tgname = 'update_short_term_search_vector_trigger'
                        ) THEN
                            CREATE TRIGGER update_short_term_search_vector_trigger
                            BEFORE INSERT OR UPDATE ON short_term_memory
                            FOR EACH ROW EXECUTE FUNCTION update_short_term_search_vector();
                        END IF;
                    END;
                    $$;
                """
                )

                conn.exec_driver_sql(
                    """
                    CREATE OR REPLACE FUNCTION update_long_term_search_vector() RETURNS trigger AS $$
                    BEGIN
                        NEW.search_vector := to_tsvector('english', COALESCE(NEW.searchable_content, '') || ' ' || COALESCE(NEW.summary, '') || ' ' || COALESCE(NEW.topic, '') || ' ' || COALESCE(NEW.symbolic_anchors, ''));
                        RETURN NEW;
                    END
                    $$ LANGUAGE plpgsql;
                """
                )

                conn.exec_driver_sql(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_trigger
                            WHERE tgname = 'update_long_term_search_vector_trigger'
                        ) THEN
                            CREATE TRIGGER update_long_term_search_vector_trigger
                            BEFORE INSERT OR UPDATE ON long_term_memory
                            FOR EACH ROW EXECUTE FUNCTION update_long_term_search_vector();
                        END IF;
                    END;
                    $$;
                """
                )

                conn.commit()
            except Exception:
                # Extensions or functions might already exist. Ensure the transaction
                # is reset so later commands don't hit InFailedSqlTransaction errors.
                conn.rollback()


def configure_sqlite_fts(engine):
    """Configure SQLite FTS5"""
    if engine.dialect.name == "sqlite":
        with engine.connect() as conn:
            try:
                # Create FTS5 virtual table for SQLite
                conn.exec_driver_sql(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_search_fts USING fts5(
                        memory_id,
                        memory_type,
                        namespace,
                        searchable_content,
                        summary,
                        category_primary,
                        symbolic_anchors,
                        content='',
                        contentless_delete=1
                    )
                """
                )

                # Create triggers to maintain FTS5 index
                conn.exec_driver_sql(
                    """
                    CREATE TRIGGER IF NOT EXISTS short_term_memory_fts_insert AFTER INSERT ON short_term_memory
                    BEGIN
                        INSERT INTO memory_search_fts(rowid, memory_id, memory_type, namespace, searchable_content, summary, category_primary, symbolic_anchors)
                        VALUES (NEW.rowid, NEW.memory_id, 'short_term', NEW.namespace, NEW.searchable_content, NEW.summary, NEW.category_primary, NEW.symbolic_anchors);
                    END
                """
                )

                conn.exec_driver_sql(
                    """
                    CREATE TRIGGER IF NOT EXISTS long_term_memory_fts_insert AFTER INSERT ON long_term_memory
                    BEGIN
                        INSERT INTO memory_search_fts(rowid, memory_id, memory_type, namespace, searchable_content, summary, category_primary, symbolic_anchors)
                        VALUES (NEW.rowid, NEW.memory_id, 'long_term', NEW.namespace, NEW.searchable_content, NEW.summary, NEW.category_primary, NEW.symbolic_anchors);
                    END
                """
                )

                conn.exec_driver_sql(
                    """
                    CREATE TRIGGER IF NOT EXISTS short_term_memory_fts_update AFTER UPDATE ON short_term_memory
                    BEGIN
                        DELETE FROM memory_search_fts WHERE rowid = OLD.rowid;
                        INSERT INTO memory_search_fts(rowid, memory_id, memory_type, namespace, searchable_content, summary, category_primary, symbolic_anchors)
                        VALUES (NEW.rowid, NEW.memory_id, 'short_term', NEW.namespace, NEW.searchable_content, NEW.summary, NEW.category_primary, NEW.symbolic_anchors);
                    END
                """
                )

                conn.exec_driver_sql(
                    """
                    CREATE TRIGGER IF NOT EXISTS long_term_memory_fts_update AFTER UPDATE ON long_term_memory
                    BEGIN
                        DELETE FROM memory_search_fts WHERE rowid = OLD.rowid;
                        INSERT INTO memory_search_fts(rowid, memory_id, memory_type, namespace, searchable_content, summary, category_primary, symbolic_anchors)
                        VALUES (NEW.rowid, NEW.memory_id, 'long_term', NEW.namespace, NEW.searchable_content, NEW.summary, NEW.category_primary, NEW.symbolic_anchors);
                    END
                """
                )

                conn.exec_driver_sql(
                    """
                    CREATE TRIGGER IF NOT EXISTS short_term_memory_fts_delete AFTER DELETE ON short_term_memory
                    BEGIN
                        DELETE FROM memory_search_fts WHERE rowid = OLD.rowid;
                    END
                """
                )

                conn.exec_driver_sql(
                    """
                    CREATE TRIGGER IF NOT EXISTS long_term_memory_fts_delete AFTER DELETE ON long_term_memory
                    BEGIN
                        DELETE FROM memory_search_fts WHERE rowid = OLD.rowid;
                    END
                """
                )

                conn.commit()
            except Exception:
                # FTS5 might not be available
                pass


class DatabaseManager:
    """SQLAlchemy-based database manager for cross-database compatibility"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            json_serializer=self._json_serializer,
            json_deserializer=self._json_deserializer,
            echo=False,  # Set to True for SQL debugging
        )

        # Configure database-specific features
        self._setup_database_features()

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _json_serializer(self, obj):
        """Custom JSON serializer"""
        import json

        return json.dumps(obj, default=str, ensure_ascii=False)

    def _json_deserializer(self, value):
        """Custom JSON deserializer"""
        import json

        return json.loads(value)

    def _setup_database_features(self):
        """Setup database-specific features like full-text search"""
        dialect_name = self.engine.dialect.name

        if dialect_name == "mysql":
            configure_mysql_fulltext(self.engine)
        elif dialect_name == "postgresql":
            configure_postgresql_fts(self.engine)
        elif dialect_name == "sqlite":
            configure_sqlite_fts(self.engine)

    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)

        # Setup database-specific search features after table creation
        self._setup_database_features()

    def get_session(self):
        """Get database session"""
        return self.SessionLocal()

    def get_database_info(self) -> dict[str, Any]:
        """Get database information"""
        return {
            "database_type": self.engine.dialect.name,
            "database_url": (
                self.database_url.split("@")[-1]
                if "@" in self.database_url
                else self.database_url
            ),
            "driver": self.engine.dialect.driver,
            "server_version": getattr(self.engine.dialect, "server_version_info", None),
        }
