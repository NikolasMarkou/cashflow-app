"""Net External Cash Flow (NECF) schema - SDD Sections 9-10."""

from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class NECFRecord(BaseModel):
    """Monthly Net External Cash Flow record.

    NECF represents the monthly aggregated external cash flow after
    internal transfer netting (SDD Section 9.4).
    """

    customer_id: str = Field(..., description="Customer identifier")
    month_key: str = Field(..., pattern=r"^\d{4}-\d{2}$", description="YYYY-MM format")
    necf: float = Field(..., description="Net External Cash Flow for the month")
    credit_total: float = Field(default=0.0, description="Total credits (inflows)")
    debit_total: float = Field(default=0.0, description="Total debits (outflows)")
    transaction_count: int = Field(default=0, ge=0, description="Number of external transactions")
    transfers_netted_count: int = Field(
        default=0, ge=0, description="Number of internal transfers removed"
    )
    transfers_netted_volume: float = Field(
        default=0.0, description="Total volume of netted transfers"
    )


class DecomposedNECF(BaseModel):
    """NECF decomposed into deterministic and residual components.

    Per SDD Section 10.2:
    NECF = Deterministic Base + Residual

    The deterministic base includes predictable flows (salary, loans, recurring).
    The residual represents behavioral volatility (discretionary spending).
    """

    customer_id: str = Field(..., description="Customer identifier")
    month_key: str = Field(..., pattern=r"^\d{4}-\d{2}$", description="YYYY-MM format")
    necf: float = Field(..., description="Original NECF value")
    deterministic_base: float = Field(
        ..., description="Sum of predictable recurring flows"
    )
    residual: float = Field(..., description="NECF - deterministic_base")

    # Outlier treatment fields (dual-value model per SDD Section 11.4)
    residual_original: Optional[float] = Field(
        None, description="Original residual before outlier treatment"
    )
    residual_clean: Optional[float] = Field(
        None, description="Residual after outlier treatment (for modeling)"
    )
    is_outlier: bool = Field(default=False, description="Whether residual was flagged as outlier")
    outlier_score: Optional[float] = Field(None, description="Outlier detection score (e.g., MZ-Score)")
    treatment_tag: Optional[str] = Field(
        None, description="NORMAL or ABNORMAL_EXTERNAL_FLOW"
    )

    @computed_field
    @property
    def decomposition_valid(self) -> bool:
        """Verify integrity constraint: NECF = Deterministic + Residual."""
        epsilon = 0.01  # Floating-point tolerance
        return abs(self.necf - (self.deterministic_base + self.residual)) < epsilon

    @computed_field
    @property
    def modeling_residual(self) -> float:
        """Return the residual value to use for modeling (clean if available)."""
        return self.residual_clean if self.residual_clean is not None else self.residual

    model_config = {
        "populate_by_name": True,
    }
