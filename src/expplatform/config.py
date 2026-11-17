"""Pydantic models describing an experiment (shared by the library, API and dashboard)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class MetricSpec(BaseModel):
    name: str
    kind: Literal["continuous", "binary", "ratio"] = "continuous"
    column: str | None = None  # continuous/binary: outcome column (defaults to ``name``)
    numerator: str | None = None  # ratio
    denominator: str | None = None  # ratio
    role: Literal["primary", "secondary", "guardrail"] = "secondary"
    higher_is_better: bool = True
    covariate: str | None = None  # pre-period column for CUPED
    ni_margin: float | None = Field(default=None, gt=0)  # guardrails: absolute margin

    @model_validator(mode="after")
    def _check(self) -> MetricSpec:
        if self.kind == "ratio" and not (self.numerator and self.denominator):
            raise ValueError("ratio metrics need numerator and denominator columns")
        if self.role == "guardrail" and self.ni_margin is None:
            raise ValueError("guardrail metrics need ni_margin")
        return self

    @property
    def col(self) -> str:
        return self.column or self.name


class SequentialSpec(BaseModel):
    enabled: bool = True
    looks: int = Field(default=20, ge=2, le=200)
    mixture_sd: float | None = Field(default=None, gt=0)  # tau; default 0.1 * metric sd
    planned_gs_looks: int = Field(default=5, ge=2, le=20)


class ExperimentConfig(BaseModel):
    name: str
    salt: str = "default-salt"
    variants: dict[str, float] = Field(default_factory=lambda: {"control": 0.5, "treatment": 0.5})
    control: str = "control"
    alpha: float = Field(default=0.05, gt=0, lt=1)
    unit_col: str = "unit_id"
    variant_col: str = "variant"
    time_col: str = "ts"
    metrics: list[MetricSpec]
    correction: Literal["holm", "bh", "none"] = "holm"
    sequential: SequentialSpec = Field(default_factory=SequentialSpec)
    segment_col: str | None = None
    srm_threshold: float = 0.001

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        if self.control not in self.variants or len(self.variants) != 2:
            raise ValueError("exactly two variants (control + one treatment) are supported")
        if sum(1 for m in self.metrics if m.role == "primary") != 1:
            raise ValueError("exactly one primary metric is required")
        return self

    @property
    def treatment(self) -> str:
        return next(k for k in self.variants if k != self.control)
