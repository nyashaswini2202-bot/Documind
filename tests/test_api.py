import json
from pathlib import Path

HEADERS = {"X-Workspace-Id": "testworkspace0001"}
POLICY = (Path(__file__).resolve().parent.parent / "sample_docs" / "remote_work_policy.txt").read_bytes()


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        name, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if data:
            events.append((name, json.loads(data)))
    return events


def test_health_and_config(client):
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["llm"]["available"] is False and "hybrid" in cfg["modes"]


def test_requires_workspace_header(client):
    assert client.get("/api/documents").status_code == 400


def test_upload_chat_delete_flow(client):
    files = [("files", ("policy.txt", POLICY, "text/plain"))]
    r = client.post("/api/documents", files=files, headers=HEADERS)
    assert r.status_code == 200 and r.json()["results"][0]["status"] == "indexed"
    doc_id = r.json()["documents"][0]["id"]

    dup = client.post("/api/documents", files=files, headers=HEADERS).json()
    assert dup["results"][0]["status"] == "duplicate"

    bad = client.post("/api/documents", files=[("files", ("x.exe", b"abc", "application/octet-stream"))], headers=HEADERS).json()
    assert bad["results"][0]["status"] == "error"

    chat = client.post("/api/chat", json={"question": "How many remote days per week are allowed?", "mode": "hybrid"}, headers=HEADERS)
    events = parse_sse(chat.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and "token" in kinds
    sources = events[0][1]["sources"]
    assert sources and "three days per week" in sources[0]["text"]

    assert client.delete(f"/api/documents/{doc_id}", headers=HEADERS).json()["documents"] == []
    empty = parse_sse(client.post("/api/chat", json={"question": "anything"}, headers=HEADERS).text)
    assert empty[0][0] == "token"


def test_evaluation_requires_llm(client):
    client.post("/api/documents", files=[("files", ("policy.txt", POLICY, "text/plain"))], headers=HEADERS)
    r = client.post("/api/evaluate", json={"num_questions": 3}, headers=HEADERS)
    assert r.status_code == 400
    client.delete("/api/documents", headers=HEADERS)
