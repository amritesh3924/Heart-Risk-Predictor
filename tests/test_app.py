import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import pytest

from app import app, db, calculate_bmi, get_risk_level, validate_input

VALID = {"age": 50, "gender": 1, "height": 170, "weight": 70, "ap_hi": 120,
         "ap_lo": 80, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.app_context():
        db.drop_all()
        db.create_all()
    with app.test_client() as c:
        yield c


def register(client, email="a@b.com"):
    return client.post("/register", data={"name": "Test", "email": email, "password": "pass123"})


# ---- Unit tests ----
def test_tc01_validate_missing_field():
    data = dict(VALID)
    del data["age"]
    assert validate_input(data) == "Missing field: age"


def test_tc02_validate_non_numeric():
    assert validate_input({**VALID, "weight": "abc"}) == "Invalid value for weight"


def test_tc03_bmi_calculation():
    assert calculate_bmi(170, 70) == 24.22


def test_tc04_risk_level_thresholds():
    assert get_risk_level(39.99) == "Low"
    assert get_risk_level(40) == "Moderate"
    assert get_risk_level(70) == "High"


# ---- API tests ----
def test_tc05_register_redirects(client):
    r = register(client)
    assert r.status_code == 302


def test_tc06_login_valid(client):
    register(client)
    client.get("/logout")
    r = client.post("/login", data={"email": "a@b.com", "password": "pass123"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/predict-page")


def test_tc07_login_wrong_password(client):
    register(client)
    client.get("/logout")
    r = client.post("/login", data={"email": "a@b.com", "password": "wrong"})
    assert r.status_code == 200


def test_tc08_predict_requires_login(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 302


def test_tc09_predict_valid(client):
    register(client)
    r = client.post("/predict", json=VALID)
    body = r.get_json()
    assert r.status_code == 200
    assert 0 <= body["risk_percent"] <= 100
    assert body["risk_level"] in ("Low", "Moderate", "High")
    assert body["bmi"] == 24.22


def test_tc10_predict_missing_fields(client):
    register(client)
    r = client.post("/predict", json={"age": 50})
    assert r.status_code == 400


def test_tc11_history_requires_login(client):
    assert client.get("/history").status_code == 302


def test_tc12_pdf_report(client):
    register(client)
    client.post("/predict", json=VALID)
    r = client.get("/report/1")
    assert r.status_code == 200 and r.mimetype == "application/pdf"


def test_tc13_report_of_other_user_forbidden(client):
    register(client, "a@b.com")
    client.post("/predict", json=VALID)
    client.get("/logout")
    register(client, "c@d.com")
    assert client.get("/report/1").status_code == 403
