"""
Recovery Router for RecoverAI REST API.

Exposes endpoints to trigger the LangGraph recovery workflow and query in-memory audit logs.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from pydantic import ValidationError

from backend.schemas.recovery import (
    TransactionInputSchema,
    RecoveryTriggerResponse,
    ErrorResponse
)
from agent.graph import run_agent
from agent.demo_data import build_test_transaction
from database.database import get_db
from database.repository import save_recovery_audit, get_audit_by_transaction_id

router = APIRouter()

# In-memory audit store for active session fallback
_IN_MEMORY_AUDIT_STORE: Dict[str, Dict[str, Any]] = {}


def get_in_memory_audit_store() -> Dict[str, Dict[str, Any]]:
    """Returns reference to in-memory audit storage."""
    return _IN_MEMORY_AUDIT_STORE


@router.post(
    "/trigger",
    response_model=RecoveryTriggerResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger Revenue Recovery Workflow",
    description="Accepts a failed payment event, executes ML prediction, failure diagnosis, policy guard evaluation, recovery action, and persists audit logs to PostgreSQL."
)
async def trigger_recovery(
    payload: TransactionInputSchema,
    db: Session = Depends(get_db)
):
    try:
        # Construct complete transaction dict using factory to guarantee ML feature schema
        payload_dict = payload.model_dump(exclude_unset=True)
        txn_dict = build_test_transaction(**payload_dict)

        # Invoke LangGraph agent workflow
        result = run_agent(txn_dict)

        # 1. Persist to PostgreSQL / Database via Repository
        try:
            save_recovery_audit(db, result)
        except Exception as db_err:
            print(f"[Warning] Failed to persist audit record to database: {db_err}")

        # 2. Store audit log in active session memory store as fallback
        if result.get("transaction_id") and result.get("audit_event"):
            _IN_MEMORY_AUDIT_STORE[result["transaction_id"]] = result["audit_event"]

        return result

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "status": "error",
                "message": str(ve),
                "error_code": "INVALID_TRANSACTION_PAYLOAD"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "An unexpected failure occurred during revenue recovery processing.",
                "error_code": "AGENT_EXECUTION_ERROR"
            }
        )


@router.get(
    "/audit/{transaction_id}",
    status_code=status.HTTP_200_OK,
    summary="Get Recovery Audit Log",
    description="Retrieves the structured audit record for a given transaction ID from PostgreSQL database (or session memory fallback)."
)
async def get_audit_log(
    transaction_id: str,
    db: Session = Depends(get_db)
):
    # 1. Try querying primary database store
    try:
        db_audit = get_audit_by_transaction_id(db, transaction_id)
        if db_audit:
            return db_audit
    except Exception as db_err:
        print(f"[Warning] Database query error for transaction {transaction_id}: {db_err}")

    # 2. Fall back to active in-memory session store
    if transaction_id in _IN_MEMORY_AUDIT_STORE:
        return _IN_MEMORY_AUDIT_STORE[transaction_id]

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "status": "error",
            "message": f"Audit record for transaction '{transaction_id}' not found in database or session store.",
            "error_code": "AUDIT_RECORD_NOT_FOUND"
        }
    )
