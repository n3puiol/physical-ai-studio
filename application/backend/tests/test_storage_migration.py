from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import storage_migration
from db.schema import Base, DatasetDB, ModelDB, ProjectDB, ProjectEnvironmentDB, ProjectRobotDB, SnapshotDB
from settings import Settings
from storage_migration import StorageMigrationError, migrate_default_storage_dir


@pytest.fixture(autouse=True)
def isolate_migration_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTO_MIGRATE_STORAGE_DIR", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("STORAGE_DIR", raising=False)
    monkeypatch.setattr(storage_migration, "OLD_DEFAULT_DATA_DIR_CANDIDATES", (tmp_path / "missing-data",))


def _settings(tmp_path: Path) -> Settings:
    return Settings(STORAGE_DIR=str(tmp_path / "new-storage"))


def _old_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    old_storage = tmp_path / "old-cache" / "physicalai"
    monkeypatch.setattr(storage_migration, "OLD_DEFAULT_STORAGE_DIR", old_storage)
    return old_storage


def _old_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    old_data_dir = tmp_path / "old-data"
    monkeypatch.setattr(storage_migration, "OLD_DEFAULT_DATA_DIR_CANDIDATES", (old_data_dir,))
    return old_data_dir


def test_migration_noops_when_old_storage_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    _old_storage(tmp_path, monkeypatch)

    migrate_default_storage_dir(settings, interactive=False)

    assert not settings.storage_dir.exists()


def test_migration_skips_explicit_storage_dir_without_auto_migrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    old_storage.mkdir(parents=True)
    (old_storage / "datasets").mkdir()
    monkeypatch.setenv("STORAGE_DIR", str(settings.storage_dir))
    monkeypatch.delenv("AUTO_MIGRATE_STORAGE_DIR", raising=False)

    migrate_default_storage_dir(settings, interactive=False)

    assert old_storage.exists()
    assert not settings.storage_dir.exists()


def test_migration_moves_old_storage_to_missing_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    (old_storage / "datasets").mkdir(parents=True)
    (old_storage / "datasets" / "dataset.txt").write_text("data")
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    migrate_default_storage_dir(settings, interactive=False)

    assert not old_storage.exists()
    assert (settings.storage_dir / "datasets" / "dataset.txt").read_text() == "data"


def test_migration_replaces_empty_storage_scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    (old_storage / "models").mkdir(parents=True)
    (old_storage / "models" / "model.txt").write_text("model")
    for subdir in storage_migration.EXPECTED_STORAGE_SUBDIRS:
        (settings.storage_dir / subdir).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    migrate_default_storage_dir(settings, interactive=False)

    assert (settings.storage_dir / "models" / "model.txt").read_text() == "model"


def test_migration_fails_when_destination_contains_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    (old_storage / "datasets").mkdir(parents=True)
    (old_storage / "datasets" / "dataset.txt").write_text("old data")
    (old_storage / "isolate").mkdir()
    settings.storage_dir.mkdir(parents=True)
    (settings.storage_dir / "existing.txt").write_text("data")
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    with pytest.raises(StorageMigrationError, match="already contains files") as exc_info:
        migrate_default_storage_dir(settings, interactive=False)

    assert f"STORAGE_DIR={old_storage}" not in str(exc_info.value)
    assert (old_storage / "datasets" / "dataset.txt").read_text() == "old data"
    assert (settings.storage_dir / "existing.txt").read_text() == "data"


def _runtime_isolate_cache(old_storage: Path) -> Path:
    isolate_env = old_storage / "isolate" / "abc123"
    (isolate_env / "venv").mkdir(parents=True)
    (isolate_env / "project").mkdir()
    (isolate_env / ".physicalai-isolate.json").write_text("{}")
    (isolate_env / ".physicalai-engine.json").write_text("{}")
    return isolate_env


def test_migration_ignores_runtime_isolate_cache_when_destination_contains_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    isolate_env = _runtime_isolate_cache(old_storage)
    (settings.storage_dir / "datasets").mkdir(parents=True)
    (settings.storage_dir / "datasets" / "dataset.txt").write_text("data")

    migrate_default_storage_dir(settings, interactive=False)

    assert (isolate_env / ".physicalai-isolate.json").exists()
    assert (settings.storage_dir / "datasets" / "dataset.txt").read_text() == "data"


@pytest.mark.parametrize("auto_migrate", [False, True])
def test_migration_noops_when_old_storage_has_only_runtime_cache_and_empty_scaffold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, auto_migrate: bool
) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    isolate_env = _runtime_isolate_cache(old_storage)
    for subdir in storage_migration.EXPECTED_STORAGE_SUBDIRS:
        (old_storage / subdir).mkdir()
    if auto_migrate:
        monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    migrate_default_storage_dir(settings, interactive=False)

    assert (isolate_env / "venv").is_dir()
    assert not settings.storage_dir.exists()


def test_migration_leaves_runtime_isolate_cache_in_old_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    isolate_env = _runtime_isolate_cache(old_storage)
    (old_storage / "datasets").mkdir()
    (old_storage / "datasets" / "dataset.txt").write_text("data")
    (old_storage / "logs").mkdir()
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    migrate_default_storage_dir(settings, interactive=False)

    assert (settings.storage_dir / "datasets" / "dataset.txt").read_text() == "data"
    assert (settings.storage_dir / "logs").is_dir()
    assert not (settings.storage_dir / "isolate").exists()
    assert not (old_storage / "datasets").exists()
    assert not (old_storage / "logs").exists()
    assert (isolate_env / ".physicalai-engine.json").exists()


def test_migration_fails_non_interactive_without_auto_migrate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    (old_storage / "models").mkdir(parents=True)
    (old_storage / "models" / "model.txt").write_text("model")
    monkeypatch.delenv("AUTO_MIGRATE_STORAGE_DIR", raising=False)

    with pytest.raises(StorageMigrationError, match="non-interactive"):
        migrate_default_storage_dir(settings, interactive=False)

    assert (old_storage / "models" / "model.txt").exists()


def test_migration_rewrites_database_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    (old_storage / "datasets" / "dataset-1").mkdir(parents=True)
    legacy_data_dir = tmp_path / "data"
    legacy_data_dir.mkdir(parents=True)
    monkeypatch.setattr(storage_migration, "OLD_DEFAULT_DATA_DIR_CANDIDATES", (legacy_data_dir,))
    monkeypatch.setenv("DATA_DIR", str(legacy_data_dir))
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    engine = create_engine(f"sqlite:///{legacy_data_dir / settings.database_file}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        session.add_all(
            [
                ProjectDB(id="project-1", name="Project"),
                ProjectEnvironmentDB(id="environment-1", project_id="project-1", name="Environment"),
                ProjectRobotDB(
                    id="robot-1",
                    project_id="project-1",
                    name="Robot",
                    type="SO101_Follower",
                    payload={},
                ),
            ]
        )
        session.commit()

        session.add_all(
            [
                DatasetDB(
                    id="dataset-1",
                    name="Dataset",
                    path=str(old_storage / "datasets" / "dataset-1"),
                    project_id="project-1",
                    environment_id="environment-1",
                    default_task="",
                ),
                DatasetDB(
                    id="dataset-2",
                    name="External Dataset",
                    path=str(tmp_path / "external" / "dataset-2"),
                    project_id="project-1",
                    environment_id="environment-1",
                    default_task="",
                ),
                ModelDB(
                    id="model-1",
                    name="Model",
                    path=str(old_storage / "models" / "model-1"),
                    policy="act",
                    properties={},
                    project_id="project-1",
                ),
                SnapshotDB(
                    id="snapshot-1",
                    path=str(old_storage / "snapshots" / "snapshot-1"),
                    dataset_id="dataset-1",
                ),
            ]
        )
        session.commit()

    migrate_default_storage_dir(settings, interactive=False)

    with session_factory() as session:
        dataset_path = session.scalar(select(DatasetDB.path).where(DatasetDB.id == "dataset-1"))
        external_dataset_path = session.scalar(select(DatasetDB.path).where(DatasetDB.id == "dataset-2"))
        model_path = session.scalar(select(ModelDB.path).where(ModelDB.id == "model-1"))
        snapshot_path = session.scalar(select(SnapshotDB.path).where(SnapshotDB.id == "snapshot-1"))

    assert dataset_path == str(settings.storage_dir / "datasets" / "dataset-1")
    assert external_dataset_path == str(tmp_path / "external" / "dataset-2")
    assert model_path == str(settings.storage_dir / "models" / "model-1")
    assert snapshot_path == str(settings.storage_dir / "snapshots" / "snapshot-1")
    engine.dispose()


def test_migration_uses_data_dir_env_as_legacy_database_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    _old_storage(tmp_path, monkeypatch)
    old_data_dir = tmp_path / "custom-data"
    old_data_dir.mkdir(parents=True)
    (old_data_dir / settings.database_file).write_text("db")
    monkeypatch.setenv("DATA_DIR", str(old_data_dir))
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    migrate_default_storage_dir(settings, interactive=False)

    assert not (old_data_dir / settings.database_file).exists()
    assert (settings.data_dir / settings.database_file).read_text() == "db"


def test_migration_moves_database_file_without_moving_legacy_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _old_storage(tmp_path, monkeypatch)
    old_data_dir = _old_data_dir(tmp_path, monkeypatch)
    old_data_dir.mkdir(parents=True)
    (old_data_dir / settings.database_file).write_text("db")
    (old_data_dir / "runtime.txt").write_text("runtime")
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    migrate_default_storage_dir(settings, interactive=False)

    assert old_data_dir.exists()
    assert not (old_data_dir / settings.database_file).exists()
    assert (old_data_dir / "runtime.txt").read_text() == "runtime"
    assert (settings.data_dir / settings.database_file).read_text() == "db"


def test_migration_fails_when_destination_database_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    _old_storage(tmp_path, monkeypatch)
    old_data_dir = _old_data_dir(tmp_path, monkeypatch)
    old_data_dir.mkdir(parents=True)
    (old_data_dir / settings.database_file).write_text("old db")
    settings.data_dir.mkdir(parents=True)
    (settings.data_dir / settings.database_file).write_text("new db")
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    with pytest.raises(StorageMigrationError, match="already exists"):
        migrate_default_storage_dir(settings, interactive=False)

    assert (old_data_dir / settings.database_file).read_text() == "old db"
    assert (settings.data_dir / settings.database_file).read_text() == "new db"


def test_database_migration_fails_non_interactive_without_auto_migrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    _old_storage(tmp_path, monkeypatch)
    old_data_dir = _old_data_dir(tmp_path, monkeypatch)
    old_data_dir.mkdir(parents=True)
    (old_data_dir / settings.database_file).write_text("db")
    monkeypatch.delenv("AUTO_MIGRATE_STORAGE_DIR", raising=False)

    with pytest.raises(StorageMigrationError, match="non-interactive"):
        migrate_default_storage_dir(settings, interactive=False)

    assert (old_data_dir / settings.database_file).exists()
    assert not (settings.data_dir / settings.database_file).exists()


def test_migration_moves_storage_before_database_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    (old_storage / "datasets").mkdir(parents=True)
    (old_storage / "datasets" / "dataset.txt").write_text("data")
    old_data_dir = _old_data_dir(tmp_path, monkeypatch)
    old_data_dir.mkdir(parents=True)
    engine = create_engine(f"sqlite:///{old_data_dir / settings.database_file}")
    Base.metadata.create_all(engine)
    engine.dispose()
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    migrate_default_storage_dir(settings, interactive=False)

    assert not old_storage.exists()
    assert (settings.storage_dir / "datasets" / "dataset.txt").read_text() == "data"
    assert not (old_data_dir / settings.database_file).exists()
    assert (settings.data_dir / settings.database_file).is_file()
