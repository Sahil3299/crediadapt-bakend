"""FastAPI service for CrediAdapt's authenticated underwriting workflow."""

from __future__ import annotations

import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from bson import ObjectId
from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)
load_dotenv(BASE_DIR / ".env")

from affordability_agent import AffordabilityAgent
from auth import (
    ACCESS_TOKEN_MINUTES,
    REFRESH_TOKEN_DAYS,
    CurrentUser,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    require_role,
    verify_password,
)
from compliance_explanation_agent import ComplianceExplanationAgentLLM
from customer_profiling_agent import CustomerProfilingAgentLLM, ExtractedProfile
from dataset_generator import generate_loan_dataset
from db import close_database, ensure_indexes, get_database
from loan_simulator_agent import LoanSimulatorAgent
from offer_discount_agent import OfferDiscountAgentLLM
from risk_ml_agent import RiskMLAgent


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Provision before accepting requests so no application is processed only
    # in memory.
    try:
        ensure_indexes()
    except Exception as exc:
        raise RuntimeError("Unable to initialize MongoDB. Check MONGODB_URI and Atlas Network Access.") from exc
    yield
    close_database()


app = FastAPI(
    title="CrediAdapt AI Backend",
    description="Authenticated loan underwriting and application persistence service",
    version="2.0.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


profiler: Optional[CustomerProfilingAgentLLM] = None
affordability: Optional[AffordabilityAgent] = None
risk_engine: Optional[RiskMLAgent] = None
offer_engine: Optional[OfferDiscountAgentLLM] = None
simulator: Optional[LoanSimulatorAgent] = None
compliance: Optional[ComplianceExplanationAgentLLM] = None


def get_agents():
    """Lazily load the existing underwriting agents once per process."""
    global profiler, affordability, risk_engine, offer_engine, simulator, compliance
    if affordability is None:
        csv_path = BASE_DIR / "loan_risk_dataset.csv"
        if not csv_path.exists():
            generate_loan_dataset()
        try:
            profiler = CustomerProfilingAgentLLM()
        except Exception:
            profiler = None
        affordability = AffordabilityAgent(max_foir=0.60, benchmark_rate=11.5)
        risk_engine = RiskMLAgent(data_path="loan_risk_dataset.csv")
        # PolicyRAG is deliberately excluded from the deploy path: its vector
        # dependencies are too large for the target hosting footprint.
        offer_engine = OfferDiscountAgentLLM(base_rate=11.5)
        simulator = LoanSimulatorAgent()
        try:
            compliance = ComplianceExplanationAgentLLM()
        except Exception:
            compliance = None
    return profiler, affordability, risk_engine, offer_engine, simulator, compliance


class LoanRequest(BaseModel):
    monthly_income: float = Field(..., gt=0)
    age: int = Field(..., ge=18, le=75)
    gender: Literal["Female", "Male", "Other"]
    house_rent: float = Field(default=0.0, ge=0)
    existing_emi: float = Field(default=0.0, ge=0)
    cibil_score: int = Field(..., ge=300, le=900)
    requested_loan: float = Field(..., gt=0)
    requested_tenure: int = Field(..., ge=6, le=84)
    early_closure_month: int = Field(default=18, ge=1, le=84)


class RegisterRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    email: str
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=2, max_length=120)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", normalized):
            raise ValueError("Enter a valid email address.")
        return normalized


class LoginRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    email: str
    password: str = Field(min_length=1, max_length=128)
    role: Literal["applicant", "officer"]

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.lower()


class AuthResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: CurrentUser


def fallback_explanation(
    is_eligible: bool, decision_type: str, requested_loan: float, recommended_loan: float,
    recommended_tenure: int, interest_rate: float, requested_foir: float, final_foir: float,
    max_foir: float, reason: str, emi: float, risk_band: str, default_prob: float,
    total_discount: float, base_rate: float, closure_month: int, outstanding_principal: float,
    interest_saved: float,
) -> dict[str, str]:
    pct = lambda value: f"{value * 100:.2f}%"
    inr = lambda value: f"₹{round(value):,}"
    if not is_eligible:
        return {
            "executive_summary": "The requested loan is not eligible based on the information provided.",
            "affordability_rationale": f"The requested loan of {inr(requested_loan)} produces a FOIR of {pct(requested_foir)}, above the maximum permitted {pct(max_foir)}. {reason}",
            "risk_and_pricing_rationale": "Risk scoring and pricing were not performed because the application did not clear the affordability check.",
            "smart_prepayment_strategy": "A prepayment plan can be prepared once an affordable loan structure is available.",
        }
    summary = (
        f"The requested loan of {inr(recommended_loan)} over {recommended_tenure} months fits within the affordability limit at an indicative rate of {interest_rate}%."
        if decision_type == "APPROVED_AS_REQUESTED"
        else f"A counter-offer of {inr(recommended_loan)} over {recommended_tenure} months is recommended at an indicative rate of {interest_rate}%."
    )
    return {
        "executive_summary": summary,
        "affordability_rationale": f"Existing obligations plus the proposed EMI of {inr(emi)} give a FOIR of {pct(final_foir)} against the maximum permitted {pct(max_foir)}.",
        "risk_and_pricing_rationale": f"The model places this application in the {risk_band} risk band with a modelled default probability of {pct(default_prob)}. Concessions applied total {total_discount}% off the {base_rate}% base rate.",
        "smart_prepayment_strategy": f"Closing in month {closure_month} leaves an outstanding principal of {inr(outstanding_principal)} and saves approximately {inr(interest_saved)} in interest.",
    }


def evaluate_application(req: LoanRequest) -> dict[str, Any]:
    """Run the existing evaluation logic and retain its frontend response shape."""
    p_agent, aff_agent, risk_agent, offer_agent, sim_agent, comp_agent = get_agents()
    assert aff_agent and risk_agent and offer_agent and sim_agent
    user_input = {
        "customer_id": "APPL-AUTO", "monthly_income": req.monthly_income, "age": req.age,
        "gender": req.gender, "house_rent": req.house_rent, "existing_emi": req.existing_emi,
        "cibil_score": req.cibil_score, "requested_loan": req.requested_loan,
        "requested_tenure": req.requested_tenure,
    }
    try:
        profile = p_agent.process(user_input) if p_agent else ExtractedProfile(**user_input)
    except Exception:
        profile = ExtractedProfile(**user_input)

    affordability_result = aff_agent.assess(profile)
    is_eligible = affordability_result.get("is_eligible", False)
    decision_type = affordability_result.get("decision_type", "REJECTED")
    adjusted_loan = affordability_result.get("adjusted_loan", 0.0)
    adjusted_tenure = affordability_result.get("adjusted_tenure", 0)
    calculated_foir = affordability_result.get("foir", 0.0)
    requested_foir = affordability_result.get("requested_foir", 0.0)
    reason = affordability_result.get("reason", "")
    eligibility_message = affordability_result.get("eligibility_message", reason)
    risk_result = risk_agent.evaluate_risk(
        profile, loan_amt=adjusted_loan, tenure=adjusted_tenure, foir=calculated_foir
    )
    risk_band = risk_result.get("risk_band", "NOT_EVALUATED")
    default_prob = risk_result.get("default_probability", 0.0)
    shap_factors = [
        {
            "feature": str(key), "label": str(key).replace("_", " ").title(),
            "impact": round(float(value), 4),
            "direction": "increases_risk" if float(value) >= 0 else "reduces_risk",
        }
        for key, value in risk_result.get("top_shap_factors", [])
    ]
    final_rate, discount_data = offer_agent.compute_discounted_rate(profile)
    policies = [
        {"policy_name": item.policy_name, "discount_percentage": item.discount_percentage, "justification": item.justification}
        for item in getattr(discount_data, "concessions", [])
    ]
    total_discount = getattr(discount_data, "total_discount", 0.0)
    simulation_result = sim_agent.generate_schedule_and_closure(
        principal=adjusted_loan, rate=final_rate if is_eligible else 11.5,
        tenure_months=adjusted_tenure, early_closure_month=req.early_closure_month,
    )
    early_closure = simulation_result.get("early_closure", {})
    outstanding_principal = early_closure.get("outstanding_principal", 0.0)
    interest_saved = early_closure.get("interest_saved", 0.0)
    emi = simulation_result.get("emi", 0.0)
    explanation: dict[str, str] | None = None
    if comp_agent:
        try:
            report = comp_agent.format_final_decision(
                profile, affordability_result, risk_result, final_rate, discount_data, simulation_result
            )
            explanation = {
                "executive_summary": report.executive_summary,
                "affordability_rationale": report.affordability_rationale,
                "risk_and_pricing_rationale": report.risk_and_pricing_rationale,
                "smart_prepayment_strategy": report.smart_prepayment_strategy,
            }
        except Exception:
            explanation = None
    explanation = explanation or fallback_explanation(
        is_eligible, decision_type, req.requested_loan, adjusted_loan, adjusted_tenure,
        final_rate if is_eligible else 0.0, requested_foir, calculated_foir, 0.60, reason,
        emi, risk_band, default_prob, total_discount, 11.5, req.early_closure_month,
        outstanding_principal, interest_saved,
    )
    total_interest = simulation_result.get("total_interest", 0.0)
    total_cost = simulation_result.get("total_cost", 0.0)
    return {
        "applicant": {"customer_id": profile.customer_id, "monthly_income": profile.monthly_income, "age": profile.age, "gender": profile.gender, "house_rent": profile.house_rent, "existing_emi": profile.existing_emi, "cibil_score": profile.cibil_score},
        "requested": {"loan_amount": req.requested_loan, "tenure_months": req.requested_tenure, "early_closure_month": req.early_closure_month},
        "decision_type": decision_type, "is_eligible": is_eligible, "eligibility_message": eligibility_message,
        "reason": reason, "requested_foir": round(requested_foir, 4), "final_foir": round(calculated_foir, 4),
        "max_foir": 0.60, "recommended_loan": adjusted_loan, "recommended_tenure": adjusted_tenure,
        "interest_rate": final_rate if is_eligible else 0.0, "base_rate": 11.5, "emi": emi,
        "risk": {"evaluated": is_eligible, "risk_band": risk_band, "default_probability": default_prob, "shap_factors": shap_factors, "message": risk_result.get("message")},
        "policies": policies, "total_discount": total_discount,
        "simulation": {
            "is_simulatable": simulation_result.get("is_simulatable", is_eligible), "emi": emi,
            "total_interest": total_interest, "total_cost": total_cost, "total_payment": total_cost,
            "schedule_length": adjusted_tenure, "closure_month": req.early_closure_month if is_eligible else 0,
            "outstanding_principal": outstanding_principal,
            "interest_paid_until_closure": early_closure.get("interest_paid_until_closure", 0.0),
            "interest_saved": interest_saved,
            "total_repaid_with_closure": early_closure.get("total_repaid", 0.0),
            "message": simulation_result.get("message", "Early closure simulation calculated successfully."),
            "schedule_preview": [],
        },
        "explanation": explanation, "engine_source": "python_multi_agent",
    }


def public_user(user: dict) -> CurrentUser:
    return CurrentUser(id=str(user["_id"]), email=user["email"], full_name=user["full_name"], role=user["role"])


def auth_response(user: dict) -> AuthResponse:
    return AuthResponse(access_token=create_access_token(user), expires_in=ACCESS_TOKEN_MINUTES * 60, user=public_user(user))


def set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key="crediadapt_refresh", value=token, httponly=True,
        secure=os.getenv("COOKIE_SECURE", "false").lower() == "true", samesite="lax",
        max_age=REFRESH_TOKEN_DAYS * 24 * 60 * 60, path="/api/auth",
    )


def audit(action: str, application_id: str | None, actor: CurrentUser, details: dict[str, Any] | None = None) -> None:
    get_database().audit_logs.insert_one({
        "timestamp": datetime.now(UTC), "actor_id": actor.id, "actor_role": actor.role,
        "action": action, "application_id": application_id, "details": details or {},
        "model_version": "1.0.0", "policy_version": "1.0.0",
    })


def next_reference() -> str:
    year = datetime.now(UTC).year
    counter = get_database().counters.find_one_and_update(
        {"_id": f"application_reference_{year}"}, {"$inc": {"value": 1}}, upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return f"CA-{year}-{counter['value']:06d}"


def get_application(application_id: str) -> dict:
    if not ObjectId.is_valid(application_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found.")
    document = get_database().applications.find_one({"_id": ObjectId(application_id)})
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found.")
    return document


def check_application_access(application: dict, user: CurrentUser) -> None:
    if user.role == "applicant" and application["applicant_id"] != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this application.")


def serialize_application(document: dict) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "_id"} | {"id": str(document["_id"])}


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "CrediAdapt AI Python Backend", "version": "2.0.0", "agents_count": 6}


@app.post("/api/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, response: Response):
    user = {
        "email": payload.email, "password_hash": hash_password(payload.password), "full_name": payload.full_name,
        "role": "applicant", "created_at": datetime.now(UTC), "is_active": True,
    }
    try:
        inserted = get_database().users.insert_one(user)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.") from exc
    user["_id"] = inserted.inserted_id
    current = public_user(user)
    audit("applicant_registered", None, current)
    set_refresh_cookie(response, create_refresh_token(user))
    return auth_response(user)


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, response: Response):
    user = get_database().users.find_one({"email": payload.email})
    if not user or not user.get("is_active") or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    if user["role"] != payload.role:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"This account is registered as an {user['role']}. Use the matching sign-in page.")
    current = public_user(user)
    audit("user_logged_in", None, current)
    set_refresh_cookie(response, create_refresh_token(user))
    return auth_response(user)


@app.post("/api/auth/refresh", response_model=AuthResponse)
def refresh(response: Response, crediadapt_refresh: str | None = Cookie(default=None)):
    if not crediadapt_refresh:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token is required.")
    payload = decode_token(crediadapt_refresh, "refresh")
    if not ObjectId.is_valid(payload["sub"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token.")
    user = get_database().users.find_one({"_id": ObjectId(payload["sub"]), "is_active": True})
    if not user or user["role"] != payload["role"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token is no longer valid.")
    set_refresh_cookie(response, create_refresh_token(user))
    return auth_response(user)


@app.post("/api/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response):
    response.delete_cookie("crediadapt_refresh", path="/api/auth")


@app.get("/api/auth/me", response_model=CurrentUser)
def me(current_user: Annotated[CurrentUser, Depends(get_current_user)]):
    return current_user


@app.post("/api/evaluate-loan")
def evaluate_loan(req: LoanRequest, _: Annotated[CurrentUser, Depends(get_current_user)]):
    """Authenticated non-persistent evaluation retained for API compatibility."""
    return evaluate_application(req)


@app.post("/api/recommendations", status_code=status.HTTP_201_CREATED)
def create_recommendation(req: LoanRequest, current_user: Annotated[CurrentUser, Depends(require_role("applicant"))]):
    result = evaluate_application(req)
    now = datetime.now(UTC)
    history = [{"from": None, "to": "SUBMITTED", "by": current_user.id, "role": current_user.role, "reason": "Application submitted", "at": now}]
    application = {
        "reference": next_reference(), "applicant_id": current_user.id, "input": req.model_dump(),
        "result": result, "status": "SUBMITTED", "status_history": history,
        "assigned_officer_id": None, "created_at": now, "updated_at": now,
    }
    inserted = get_database().applications.insert_one(application)
    application["_id"] = inserted.inserted_id
    audit("application_submitted", str(inserted.inserted_id), current_user, {"reference": application["reference"]})
    return serialize_application(application)


@app.get("/api/recommendations/{application_id}")
def recommendation(application_id: str, current_user: Annotated[CurrentUser, Depends(get_current_user)]):
    application = get_application(application_id)
    check_application_access(application, current_user)
    return serialize_application(application)


@app.get("/api/applications")
def applications(current_user: Annotated[CurrentUser, Depends(get_current_user)]):
    query = {} if current_user.role == "officer" else {"applicant_id": current_user.id}
    documents = get_database().applications.find(query).sort("created_at", -1)
    return {"items": [serialize_application(document) for document in documents]}


@app.get("/api/officer/queue")
def officer_queue(current_user: Annotated[CurrentUser, Depends(require_role("officer"))]):
    """Phase 1 role-protected queue; filters and review actions follow in Phase 2."""
    documents = get_database().applications.find({}).sort("created_at", -1)
    return {"items": [serialize_application(document) for document in documents]}


@app.get("/api/explainability/{application_id}")
def explainability(application_id: str, current_user: Annotated[CurrentUser, Depends(get_current_user)]):
    application = get_application(application_id)
    check_application_access(application, current_user)
    return {"application_id": application_id, "reference": application["reference"], "risk": application["result"]["risk"], "explanation": application["result"]["explanation"]}


@app.get("/api/advisor/{application_id}")
def advisor(application_id: str, current_user: Annotated[CurrentUser, Depends(get_current_user)]):
    application = get_application(application_id)
    check_application_access(application, current_user)
    return {"application_id": application_id, "reference": application["reference"], "advisor": application["result"]["explanation"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
