import io

import pytest
from fastapi.testclient import TestClient

from expplatform.api import create_app
from expplatform.demo import demo_config, demo_frame
from expplatform.report import build_readout


@pytest.fixture()
def client():
    return TestClient(create_app())


@pytest.fixture(scope="module")
def frame():
    return demo_frame(seed=3, n=12_000)


def _create(client):
    r = client.post("/experiments", json=demo_config().model_dump())
    assert r.status_code == 201
    return r.json()["id"]


def _csv(df):
    return df.to_csv(index=False).encode()


def test_health_and_sample_size(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/design/sample-size", json={"kind": "means", "delta": 0.2, "sd": 1.0})
    assert r.json() == {"n_control": 393, "n_treatment": 393, "total": 786}
    r = client.post("/design/sample-size",
                    json={"kind": "proportions", "p_control": 0.1, "p_treatment": 0.12, "r2": 0.2})
    assert r.status_code == 200 and r.json()["n_control"] < 3880
    assert client.post("/design/sample-size", json={"kind": "means", "delta": 0.2}).status_code == 422


def test_create_list_get_and_validation(client):
    exp_id = _create(client)
    assert client.get("/experiments").json()[0]["id"] == exp_id
    assert client.get(f"/experiments/{exp_id}").json()["config"]["name"] == "checkout-redesign"
    assert client.get("/experiments/nope").status_code == 404
    bad = demo_config().model_dump()
    bad["metrics"] = [m for m in bad["metrics"] if m["role"] != "primary"]
    assert client.post("/experiments", json=bad).status_code == 422


def test_assign_endpoint_is_deterministic(client):
    exp_id = _create(client)
    a = client.get(f"/experiments/{exp_id}/assign/user-7").json()
    b = client.get(f"/experiments/{exp_id}/assign/user-7").json()
    assert a == b and a["variant"] in ("control", "treatment")


def test_csv_ingest_and_analysis_matches_library(client, frame):
    exp_id = _create(client)
    r = client.post(f"/experiments/{exp_id}/events?format=csv", content=_csv(frame))
    assert r.json() == {"ingested": len(frame), "total_units": len(frame)}
    out = client.get(f"/experiments/{exp_id}/analysis").json()
    assert out["status"] in {"ship_candidate", "inconclusive", "hold_guardrail"}
    assert out["srm"]["mismatch"] is False and out["assignment_check"]["ok"]
    direct = build_readout(frame, demo_config())
    primary = next(m for m in out["metrics"] if m["role"] == "primary")
    ref = next(m for m in direct["metrics"] if m["role"] == "primary")
    assert primary["effect"]["estimate"] == pytest.approx(ref["effect"]["estimate"], rel=1e-9)
    assert primary["variance_reduction"] > 0.15
    assert out["sequential"]["enabled"] and len(out["sequential"]["n"]) >= 10
    assert out["guardrails"][0]["name"] == "latency_ms"
    # re-posting the same units is idempotent
    r2 = client.post(f"/experiments/{exp_id}/events?format=csv", content=_csv(frame))
    assert r2.json()["total_units"] == len(frame)


def test_parquet_ingest(client, frame):
    exp_id = _create(client)
    buf = io.BytesIO()
    frame.to_parquet(buf)
    r = client.post(f"/experiments/{exp_id}/events?format=parquet", content=buf.getvalue())
    assert r.status_code == 200 and r.json()["total_units"] == len(frame)
    assert client.get(f"/experiments/{exp_id}/analysis").status_code == 200


def test_srm_bug_surfaces_in_analysis(client):
    bad = demo_frame(seed=3, n=12_000, srm_drop=0.06)
    exp_id = _create(client)
    client.post(f"/experiments/{exp_id}/events?format=csv", content=_csv(bad))
    out = client.get(f"/experiments/{exp_id}/analysis").json()
    assert out["srm"]["mismatch"] and out["status"] == "invalid_srm"
    assert any("mismatch" in w.lower() for w in out["warnings"])


def test_ingest_errors(client, frame):
    exp_id = _create(client)
    assert client.get(f"/experiments/{exp_id}/analysis").status_code == 409
    assert client.post(f"/experiments/{exp_id}/events?format=csv",
                       content=b"a,b\n1,2\n").status_code == 422
    assert client.post(f"/experiments/{exp_id}/events?format=xml", content=b"x").status_code == 400
    odd = frame.head(10).assign(variant="mystery")
    assert client.post(f"/experiments/{exp_id}/events?format=csv", content=_csv(odd)).status_code == 422
    assert client.post("/experiments/nope/events", content=b"x").status_code == 404
    missing_cols = frame[["unit_id", "variant"]]
    client.post(f"/experiments/{exp_id}/events?format=csv", content=_csv(missing_cols))
    assert client.get(f"/experiments/{exp_id}/analysis").status_code == 422
