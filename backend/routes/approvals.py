"""
Human-in-the-Loop (HITL) Approvals Router for RecoverAI REST API.

Provides endpoints to:
1. List pending approval requests for high-value / policy-escalated transactions.
2. Retrieve specific approval request details.
3. Merchant approve and execute recovery action safely and idempotently.
4. Merchant reject and safely close recovery action with zero payments executed.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query, status
from sqlalchemy.orm import Session

from database.database import get_db
from database.repository import (
    list_pending_approvals,
    get_approval_request,
    approve_recovery_action,
    reject_recovery_action
)
from backend.schemas.approvals import (
    ApprovalDecisionRequest,
    ApprovalItemResponse,
    ApprovalActionResponse,
    PendingApprovalsResponse
)

router = APIRouter()


@router.get(
    "/pending",
    response_model=PendingApprovalsResponse,
    status_code=status.HTTP_200_OK,
    summary="List Pending Approval Requests",
    description="Retrieves all transactions currently waiting in the merchant human review escalation queue."
)
def get_pending_approvals(
    merchant_id: Optional[str] = Query(None, description="Optional merchant ID filter"),
    limit: int = Query(50, ge=1, le=100, description="Max records to return"),
    db: Session = Depends(get_db)
):
    items = list_pending_approvals(db=db, merchant_id=merchant_id, limit=limit)
    response_items = [ApprovalItemResponse.model_validate(item) for item in items]
    return PendingApprovalsResponse(
        total_pending=len(response_items),
        approvals=response_items
    )


@router.get(
    "/{transaction_id}",
    response_model=ApprovalItemResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Approval Request Details",
    description="Fetches complete failure diagnosis, ML scores, recommendations, and status for an approval request."
)
def get_approval_by_transaction_id(
    transaction_id: str,
    db: Session = Depends(get_db)
):
    approval_req = get_approval_request(db, transaction_id)
    if not approval_req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "NOT_FOUND",
                "message": f"Approval request for transaction '{transaction_id}' was not found."
            }
        )
    return ApprovalItemResponse.model_validate(approval_req)


@router.post(
    "/{transaction_id}/approve",
    response_model=ApprovalActionResponse,
    status_code=status.HTTP_200_OK,
    summary="Approve Recovery Action",
    description="Grants merchant authorization to execute the recovery action for an escalated transaction."
)
def approve_transaction(
    transaction_id: str,
    payload: Optional[ApprovalDecisionRequest] = None,
    db: Session = Depends(get_db)
):
    notes = payload.notes if payload else None
    dry_run = payload.dry_run if payload else False

    success, code_label, result = approve_recovery_action(
        db=db,
        transaction_id=transaction_id,
        human_notes=notes,
        dry_run=dry_run
    )

    if not success:
        if code_label == "NOT_FOUND":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result
            )
        elif code_label in ["CANNOT_APPROVE_BLOCKED", "ALREADY_REJECTED", "INVALID_STATE"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=result
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result
            )

    return ApprovalActionResponse(
        message=result.get("message", "Transaction approved"),
        transaction_id=transaction_id,
        status=result.get("status", "EXECUTED"),
        human_decision="APPROVED",
        human_notes=notes,
        execution_details=result.get("execution_details")
    )


@router.post(
    "/{transaction_id}/reject",
    response_model=ApprovalActionResponse,
    status_code=status.HTTP_200_OK,
    summary="Reject Recovery Action",
    description="Merchant rejects the escalated transaction. Safely prevents any automated recovery action."
)
def reject_transaction(
    transaction_id: str,
    payload: Optional[ApprovalDecisionRequest] = None,
    db: Session = Depends(get_db)
):
    notes = payload.notes if payload else None

    success, code_label, result = reject_recovery_action(
        db=db,
        transaction_id=transaction_id,
        human_notes=notes
    )

    if not success:
        if code_label == "NOT_FOUND":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result
            )
        elif code_label in ["ALREADY_APPROVED", "INVALID_STATE"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=result
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result
            )

    return ApprovalActionResponse(
        message=result.get("message", "Transaction rejected"),
        transaction_id=transaction_id,
        status="REJECTED_BY_HUMAN",
        human_decision="REJECTED",
        human_notes=notes,
        execution_details=None
    )
