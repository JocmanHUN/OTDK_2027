from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Money(BaseModel):
    amount: Decimal = Field(..., description="Monetary amount")
    currency: Literal["HUF", "EUR", "USD"] = Field("HUF", description="Currency code")

    model_config = ConfigDict(frozen=True)

    @field_validator("amount", mode="before")
    @classmethod
    def _ensure_decimal(cls, v: Decimal | int | float) -> Decimal:
        if not isinstance(v, Decimal):
            v = Decimal(str(v))
        return v.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)

    def _ensure_same_currency(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise ValueError("Currency mismatch")

    def __add__(self, other: "Money") -> "Money":
        self._ensure_same_currency(other)
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._ensure_same_currency(other)
        return Money(amount=self.amount - other.amount, currency=self.currency)

    def __mul__(self, factor: Decimal | int | float) -> "Money":
        amount = self.amount * (factor if isinstance(factor, Decimal) else Decimal(str(factor)))
        return Money(amount=amount, currency=self.currency)

    def ratio_to(self, other: "Money") -> Decimal:
        self._ensure_same_currency(other)
        if other.amount == 0:
            raise ZeroDivisionError("Cannot divide by zero amount")
        return self.amount / other.amount
