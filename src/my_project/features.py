"""Feature configuration for Pega ADM surrogate modelling.

Active predictors per variant are stored in VARIANT_FEATURES, keyed by pyName
(upper-case). Each entry holds the ordered feature list and the subset Pega ADM
treats as numeric. Access pattern:

    from my_project.features import VARIANT_FEATURES
    cfg = VARIANT_FEATURES["L5B15"]
    # cfg.features  → list[str] of active predictors
    # cfg.numeric   → frozenset[str] of numeric-typed predictors
"""
from __future__ import annotations

from dataclasses import dataclass

TARGET = "propensity"


@dataclass(frozen=True)
class VariantConfig:
    """Feature configuration for one Pega ADM offer-model variant."""
    features: tuple[str, ...]
    numeric:  frozenset[str]


# ── Columns to always exclude from feature matrices ────────────────────────

_ID_COLS = [
    "pxInteractionID",
    "pxSubjectID",
    "modelExecutionID",
    "TreatmentID",
    "modelVersion",
    "modelPerformance",
    "modelEvidence",
    "modelTechnique",
]

_PARTITION_COLS = [          # constant after Email/Outbound/Luggage/Sales filter
    "pyName",
    "pyChannel",
    "pyDirection",
    "pyGroup",
    "pyIssue",
]

_PII_COLS = [
    "Customer.Email",
    "Customer.WorkEmail",
    "Customer.EmailLC",
    "CustBookedFlight.Email",
    "Customer.FirstNameInternal",
    "Customer.LastNameInternal",
    "Customer.pyFirstName",
    "Customer.pyLastName",
    "Customer.pyTitle",
    "Customer.Salutation",
    "Customer.DateOfBirth",
    "Customer.FlyingBlueNumber",
    "Customer.CustomerKPIs.CustomerID",
    "CustBookedFlight.PNR",
    "CustBookedFlight.BookingRefKey",
    "Customer.ReferenceInsKey",
]

EXCLUDE_COLS = _ID_COLS + _PARTITION_COLS + _PII_COLS


# ── Per-variant feature configurations ────────────────────────────────────
# Features are listed in the order they appear in the Pega ADM snapshot.
# numeric contains the subset Pega ADM treats as numeric (all others symbolic).

VARIANT_FEATURES: dict[str, VariantConfig] = {

    # ── L5B15: 15 kg luggage add-on (primary variant) ─────────────────────
    "L5B15": VariantConfig(
        features=(
            # Booking context
            "CustBookedFlight.BookingData.BookingMonth",       # numeric
            "CustBookedFlight.BookingData.BookerGender",
            "CustBookedFlight.BookingData.CultureCode",
            "CustBookedFlight.BookingData.FlightInboundArrival",
            # Flight & passenger
            "CustBookedFlight.Language",
            "CustBookedFlight.Journey",
            "CustBookedFlight.FlightNumberOperatorIATA",
            "CustBookedFlight.SeatNumber",
            "CustBookedFlight.IsStaffStandBy",
            # Flight data
            "CustBookedFlight.FlightData.AirlineCodeIATA",
            "CustBookedFlight.FlightData.DestinationAirport",
            # Interaction history
            "IH.Email.Outbound.Pending.pyHistoricalOutcomeCount",  # numeric
            "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince",  # numeric
            "IH.Email.Outbound.Pending.pxLastGroupID",
            "IH.Email.Outbound.Delivered.pxLastOutcomeTime.DaysSince",  # numeric
            "IH.Email.Outbound.Delivered.pxLastGroupID",
            "IH.Email.Inbound.Pending.pxLastOutcomeTime.DaysSince",   # numeric
            "IH.Email.Inbound.Clicked.pxLastOutcomeTime.DaysSince",   # numeric
            "IH.Push.Outbound.Pending.pyHistoricalOutcomeCount",       # numeric
            "IH.Push.Outbound.Pending.pxLastOutcomeTime.DaysSince",    # numeric
            "IH.Push.Outbound.Pending.pxLastGroupID",
            "IH.Event.Outbound.RealTimeEvent.pyHistoricalOutcomeCount",  # numeric
            # Scoring parameters
            "param::Param.BundleName",
        ),
        numeric=frozenset({
            "CustBookedFlight.BookingData.BookingMonth",
            "IH.Email.Outbound.Pending.pyHistoricalOutcomeCount",
            "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince",
            "IH.Email.Outbound.Delivered.pxLastOutcomeTime.DaysSince",
            "IH.Email.Inbound.Pending.pxLastOutcomeTime.DaysSince",
            "IH.Email.Inbound.Clicked.pxLastOutcomeTime.DaysSince",
            "IH.Push.Outbound.Pending.pyHistoricalOutcomeCount",
            "IH.Push.Outbound.Pending.pxLastOutcomeTime.DaysSince",
            "IH.Event.Outbound.RealTimeEvent.pyHistoricalOutcomeCount",
        }),
    ),

    # ── CLUG: generic luggage (replication variant) ────────────────────────
    "CLUG": VariantConfig(
        features=(
            # Booking context
            "CustBookedFlight.BookingData.BookerGender",
            "CustBookedFlight.BookingData.BookingMonth",           # numeric
            "CustBookedFlight.BookingData.FlightInboundArrival",
            # Flight data
            "CustBookedFlight.FlightData.AirlineCodeIATA",
            "CustBookedFlight.FlightData.CommercialFlightNumberOperatorIATA",
            "CustBookedFlight.FlightData.DepartureAirport",
            "CustBookedFlight.FlightData.DestinationAirport",
            # Flight & passenger
            "CustBookedFlight.FlightNumberOperatorIATA",
            "CustBookedFlight.IsStaffStandBy",
            "CustBookedFlight.Journey",
            "CustBookedFlight.SeatNumber",
            # Interaction history
            "IH.Email.Inbound.Pending.pxLastOutcomeTime.DaysSince",    # numeric
            "IH.Email.Outbound.Clicked.pxLastGroupID",
            "IH.Email.Outbound.Delivered.pxLastGroupID",
            "IH.Email.Outbound.Pending.pxLastGroupID",
            "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince",   # numeric
            "IH.Email.Outbound.Pending.pyHistoricalOutcomeCount",       # numeric
            "IH.Event.Outbound.RealTimeEvent.pxLastOutcomeTime.DaysSince",  # numeric
            "IH.Push.Outbound.Pending.pxLastGroupID",
            "IH.Push.Outbound.Pending.pxLastOutcomeTime.DaysSince",    # numeric
            "IH.Push.Outbound.Pending.pyHistoricalOutcomeCount",        # numeric
            # Scoring parameters
            "param::Param.BundleName",
        ),
        numeric=frozenset({
            "CustBookedFlight.BookingData.BookingMonth",
            "IH.Email.Inbound.Pending.pxLastOutcomeTime.DaysSince",
            "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince",
            "IH.Email.Outbound.Pending.pyHistoricalOutcomeCount",
            "IH.Event.Outbound.RealTimeEvent.pxLastOutcomeTime.DaysSince",
            "IH.Push.Outbound.Pending.pxLastOutcomeTime.DaysSince",
            "IH.Push.Outbound.Pending.pyHistoricalOutcomeCount",
        }),
    ),
}
