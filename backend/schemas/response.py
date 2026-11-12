from typing import Literal

from pydantic import BaseModel, field_validator


def _upper(value):
    return value.upper() if isinstance(value, str) else value


class ApprovalRequest(BaseModel):
    choice: Literal["APPROVED", "REJECTED"]

    _normalise = field_validator("choice", mode="before")(_upper)


class FeedbackRequest(BaseModel):
    # A strict boolean: the old route used bool(payload[...]), which turns the string "false" into True.
    ground_truth_attack: bool
    reviewer_choice: Literal["APPROVED", "REJECTED"] | None = None

    _normalise = field_validator("reviewer_choice", mode="before")(_upper)
