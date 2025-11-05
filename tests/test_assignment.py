import os
import sys
from pathlib import Path
import types

# ensure project root (parent of tests/) is on sys.path so we can import app
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Mock the semantic_search module before importing app
fake_semantic = types.ModuleType("semantic_search")
def _fake_search(query, top_k=3):
    return [("images/fake.png", 0.999)]
fake_semantic.search = _fake_search

# Insert into sys.modules so imports see the mock.
sys.modules["semantic_search"] = fake_semantic
# also provide under "search" in case app does `import search as search_module`
sys.modules["search"] = fake_semantic

import app
from fastapi.testclient import TestClient
import pytest

def test_images_folder_exists_and_not_empty():
    assert os.path.exists('images'), "Images folder does not exist."
    assert len(os.listdir('images')) > 0, "Images folder is empty."

def test_faiss_index_file_exists():
    assert os.path.exists('faiss_index.bin'), "FAISS index file does not exist."

def test_images_folder_changed():
    import subprocess
    # Compute checksum of images folder
    result = os.popen("md5sum images/*").read()
    baseline_checksum = open("tests/baseline_images_md5").read()
    assert result != baseline_checksum, f"You need to modify the images folder"

def test_faiss_file_changed():
    import subprocess
    # Compute checksum of faiss_index.bin
    result = os.popen("md5sum faiss_index.bin").read()
    baseline_checksum = open("tests/baseline_faiss_md5").read()
    assert result != baseline_checksum, f"You need to re-create the faiss_index.bin file"    


def test_search_endpoint(monkeypatch):
    """
    Mock the search function used by the FastAPI app before calling /search.
    The app module imports the search module as `search_module`, so patch that.
    """

    def fake_search(query, top_k=3):
        # return list of (filename, score) as app expects
        return ["fake.png"]

    # Patch the search function on the module object used by app
    monkeypatch.setattr(app.search_module, "search", fake_search)

    client = TestClient(app.app)
    response = client.get("/search", params={"query": "a cat sitting on a mat"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert any("fake.png" in item for item in data)

# ensure index.html references the /search endpoint
def test_index_contains_search_endpoint():
    index_path = ROOT / "index.html"
    assert index_path.exists(), "index.html not found in project root"
    content = index_path.read_text(encoding="utf-8")
    assert "search" in content, "index.html must reference the /search endpoint"
