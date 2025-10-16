from loguru import logger

from ..config.settings import LoggingSettings, LogLevel
from ..utils.exceptions import DatabaseError
from ..utils.logging import LoggingManager


def setup_logging(self):
    """Setup logging configuration based on verbose mode"""
    if not LoggingManager.is_initialized():
        logging_settings = LoggingSettings()
        if self.verbose:
            logging_settings.level = LogLevel.DEBUG
        LoggingManager.setup_logging(logging_settings, verbose=self.verbose)
        if self.verbose:
            logger.info("Verbose logging enabled - only loguru logs will be displayed")


def setup_database(self):
    """Setup database tables based on template"""
    if not self.schema_init:
        logger.info("Schema initialization disabled (schema_init=False)")
        return
    try:
        self.db_manager.initialize_schema()
        logger.info("Database schema initialized successfully")
    except Exception as e:
        raise DatabaseError(f"Failed to setup database: {e}")
