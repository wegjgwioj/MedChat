import os
import sys
from pathlib import Path

import pytest


# Ensure repo root is on sys.path so `import app.*` works when running pytest
# from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ===== CI 环境自动配置 =====

def pytest_configure(config):
    """pytest 启动时自动配置环境变量（CI 兼容）"""
    # 如果没有设置 API 密钥，使用占位符避免导入错误
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = "test-key-for-ci"

    # 默认使用规则模式（避免 API 调用）
    if not os.environ.get("AGENT_SLOT_EXTRACTOR"):
        os.environ["AGENT_SLOT_EXTRACTOR"] = "rules"

    if not os.environ.get("CHAT_SLOT_EXTRACTOR"):
        os.environ["CHAT_SLOT_EXTRACTOR"] = "rules"


@pytest.fixture(scope="session")
def offline_mode(monkeypatch_session):
    """强制离线模式的 fixture"""
    monkeypatch_session.setenv("AGENT_SLOT_EXTRACTOR", "rules")
    monkeypatch_session.setenv("CHAT_SLOT_EXTRACTOR", "rules")


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session 级别的 monkeypatch"""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()

