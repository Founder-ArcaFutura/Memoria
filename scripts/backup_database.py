import argparse
from pathlib import Path

from loguru import logger

from memoria.config.manager import ConfigManager
from memoria.database.sqlalchemy_manager import SQLAlchemyDatabaseManager


def main() -> None:

    parser = argparse.ArgumentParser(description="Backup or restore the configured database")
    parser.add_argument("destination", help="Path to write the backup file")

    parser.add_argument(
        "--connection-string",
        dest="connection_string",
        help="Optional database connection string override",
    )
    parser.add_argument(
        "--restore",
        action="store_true",

        help="Restore the database from the backup file instead of creating one",

    )
    args = parser.parse_args()

    config = ConfigManager.get_instance()
    settings = config.get_settings()
    connection = args.connection_string or settings.database.connection_string

    manager = SQLAlchemyDatabaseManager(connection)

    destination = Path(args.destination)

    if args.restore:
        if not destination.exists():
            logger.error(f"Backup file {destination} does not exist")
            raise SystemExit(1)

        try:
            manager.restore_database(destination)
            logger.info("Database restore completed")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Database restore failed: {exc}")
            raise
    else:
        manager.backup_database(destination)

        logger.info("Database backup completed")


if __name__ == "__main__":
    main()
