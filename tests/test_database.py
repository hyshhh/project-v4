"""配置和数据库的单元测试（不需要 LLM / Embedding API）"""

import csv
import hashlib
import json
import pytest
from pathlib import Path

from config import load_config
from database import ShipDatabase


# ══════════════════════════════════════════════
#  Config 测试
# ══════════════════════════════════════════════

class TestLoadConfig:
    def test_defaults(self):
        """没有 config.yaml 时返回内置默认值"""
        c = load_config(config_path="/nonexistent/config.yaml")
        assert c["llm"]["model"] == "Qwen/Qwen3-VL-4B-AWQ"
        assert c["llm"]["api_key"] == "abc123"
        assert c["llm"]["temperature"] == 0.0
        assert c["embed"]["model"] == "Qwen3-Embedding-0.6B"
        assert c["embed"]["base_url"] == "http://localhost:7891/v1"
        assert c["retrieval"]["top_k"] == 3
        assert c["retrieval"]["score_threshold"] == 0.5
        assert c["vector_store"]["persist_path"] == "./vector_store"
        assert c["vector_store"]["auto_rebuild"] is False
        assert c["app"]["log_level"] == "INFO"
        assert c["app"]["ship_db_path"] == "./data/ships.csv"

    def test_load_from_yaml(self, tmp_path):
        """从 YAML 文件加载"""
        import yaml
        cfg_data = {
            "llm": {"model": "test-model", "api_key": "test-key"},
            "retrieval": {"top_k": 10},
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data, allow_unicode=True), encoding="utf-8")

        c = load_config(config_path=str(cfg_file))
        assert c["llm"]["model"] == "test-model"
        assert c["llm"]["api_key"] == "test-key"
        assert c["retrieval"]["top_k"] == 10
        # 未覆盖的字段不存在（不会自动合并默认值）
        assert "embed" not in c or c.get("embed", {}).get("model") != "text-embedding-v4"


# ══════════════════════════════════════════════
#  ShipDatabase 精确查找测试
# ══════════════════════════════════════════════

def _make_config(tmp_path: Path) -> dict:
    """创建测试用配置字典"""
    return {
        "embed": {"model": "text-embedding-v4", "api_key": "test", "base_url": "https://example.com/v1"},
        "retrieval": {"top_k": 3, "score_threshold": 0.5},
        "vector_store": {"persist_path": str(tmp_path / "vector_store"), "auto_rebuild": False},
        "app": {"log_level": "INFO", "ship_db_path": None},
    }


def _make_db(tmp_path: Path, csv_content: str | None = None) -> ShipDatabase:
    """创建测试用 ShipDatabase（跳过向量库构建）"""
    csv_path = tmp_path / "test_ships.csv"
    if csv_content:
        csv_path.write_text(csv_content, encoding="utf-8")
    else:
        csv_path.write_text(
            "hull_number,description\n"
            "0014,白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪\n"
            "0025,黑色散货船，船体有红色水线\n"
            "9999,测试船\n",
            encoding="utf-8",
        )

    cfg = _make_config(tmp_path)
    return ShipDatabase(config=cfg, db_path=str(csv_path))


class TestShipDatabase:
    def test_lookup_existing(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.lookup("0014") == "白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪"

    def test_lookup_missing(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.lookup("XXXX") is None

    def test_lookup_whitespace(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.lookup("  0014  ") == "白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪"

    def test_len(self, tmp_path):
        db = _make_db(tmp_path)
        assert len(db) == 3

    def test_hull_numbers(self, tmp_path):
        db = _make_db(tmp_path)
        assert "0014" in db.hull_numbers
        assert "0025" in db.hull_numbers

    def test_csv_path_property(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.csv_path.exists()
        assert db.csv_path.suffix == ".csv"

    def test_custom_csv(self, tmp_path):
        content = "hull_number,description\nA001,测试船A\nA002,测试船B\n"
        db = _make_db(tmp_path, csv_content=content)
        assert db.lookup("A001") == "测试船A"
        assert db.lookup("A002") == "测试船B"
        assert len(db) == 2

    def test_empty_csv_creates_default(self, tmp_path):
        """db_path 指向不存在的路径时，应自动创建默认 CSV"""
        cfg = _make_config(tmp_path)
        db = ShipDatabase(config=cfg, db_path=str(tmp_path / "new_dir" / "ships.csv"))
        assert db.csv_path.exists()
        assert len(db) > 0

    def test_build_documents(self, tmp_path):
        db = _make_db(tmp_path)
        docs = db._build_documents()
        assert len(docs) == 3
        for doc in docs:
            assert "弦号" in doc.page_content
            assert "hull_number" in doc.metadata
            assert "description" in doc.metadata

    def test_csv_hash_detection(self, tmp_path):
        db = _make_db(tmp_path)
        hash1 = db._compute_csv_hash()

        csv_path = db.csv_path
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write("NEW01,新添加的船\n")

        hash2 = hashlib.md5(csv_path.read_bytes()).hexdigest()
        assert hash1 != hash2

    def test_csv_changed_detection(self, tmp_path):
        db = _make_db(tmp_path)
        assert db._csv_changed() is True

    def test_save_and_load_hash(self, tmp_path):
        db = _make_db(tmp_path)
        csv_hash = db._compute_csv_hash()
        db._save_hash(csv_hash)
        loaded = db._load_saved_hash()
        assert loaded == csv_hash


class TestShipDatabaseBOM:
    def test_csv_with_bom(self, tmp_path):
        csv_path = tmp_path / "bom_ships.csv"
        content = b'\xef\xbb\xbf' + "hull_number,description\nB001,BOM测试船\n".encode("utf-8")
        csv_path.write_bytes(content)

        cfg = _make_config(tmp_path)
        db = ShipDatabase(config=cfg, db_path=str(csv_path))
        assert db.lookup("B001") == "BOM测试船"
