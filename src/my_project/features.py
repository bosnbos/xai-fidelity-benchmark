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

    # ── L5B25: 25 kg luggage add-on ───────────────────────────────────────
    # NOTE: only 584 rows in the data extract. Surrogate and explanation
    # (notebooks 04–05) are viable; stability analysis (06) is borderline —
    # temporal halves have ~292 rows each and route subgroups are too small.
    "L5B25": VariantConfig(
        features=(
            # Customer purchase history (cross-product signals)
            "Customer.CustomerKPIs.L5B20CountTotal",       # numeric
            "Customer.CustomerKPIs.CLUGCountTotal",         # numeric
            # Booking context
            "CustBookedFlight.BookingData.BookingMonth",    # numeric
            "CustBookedFlight.BookingData.BookerGender",
            "CustBookedFlight.BookingData.CultureCode",
            "CustBookedFlight.BookingData.FlightOutboundDeparture",
            # Flight & passenger
            "CustBookedFlight.IsStaffStandBy",
            "CustBookedFlight.SeatNumber",
            "CustBookedFlight.FlightNumberOperatorIATA",
            # Flight data
            "CustBookedFlight.FlightData.AircraftType",
            "CustBookedFlight.FlightData.AirlineCodeIATA",
            "CustBookedFlight.FlightData.DepartureAirport",
            # Interaction history — email outbound
            "IH.Email.Outbound.Pending.pyHistoricalOutcomeCount",   # numeric
            "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince",  # numeric
            "IH.Email.Outbound.Pending.pxLastGroupID",
            "IH.Email.Outbound.Delivered.pyHistoricalOutcomeCount",  # numeric
            # Interaction history — email inbound
            "IH.Email.Inbound.Clicked.pxLastOutcomeTime.DaysSince",  # numeric
            # Interaction history — push
            "IH.Push.Outbound.Pending.pyHistoricalOutcomeCount",     # numeric
            "IH.Push.Outbound.Pending.pxLastOutcomeTime.DaysSince",  # numeric
            "IH.Push.Outbound.Pending.pxLastGroupID",
            # Scoring parameters
            "param::Param.BundleName",
            "param::Param.Age",
        ),
        numeric=frozenset({
            "Customer.CustomerKPIs.L5B20CountTotal",
            "Customer.CustomerKPIs.CLUGCountTotal",
            "CustBookedFlight.BookingData.BookingMonth",
            "IH.Email.Outbound.Pending.pyHistoricalOutcomeCount",
            "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince",
            "IH.Email.Outbound.Delivered.pyHistoricalOutcomeCount",
            "IH.Email.Inbound.Clicked.pxLastOutcomeTime.DaysSince",
            "IH.Push.Outbound.Pending.pyHistoricalOutcomeCount",
            "IH.Push.Outbound.Pending.pxLastOutcomeTime.DaysSince",
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
    # ── BookingDotCom: hotel cross-sell (third-party) ─────────────────────────
    "BookingDotCom": VariantConfig(
        features=(
            # Scoring parameters
            "param::Param.BundleName",
            # Booking context
            "CustBookedFlight.BookingData.DepartureMonth",         # numeric
            "CustBookedFlight.FlightData.AirlineCodeIATA",
            "Customer.CultureCode",
            "CustBookedFlight.FlightData.CommercialFlightNumberOperatorIATA",
            # Interaction history — push
            "IH.Push.Outbound.Pending.pyHistoricalOutcomeCount",   # numeric
            # Flight & passenger
            "CustBookedFlight.FlightNumberOperatorIATA",
            # Interaction history — email inbound
            "IH.Email.Inbound.Pending.pxLastOutcomeTime.DaysSince",  # numeric
            # Booking context (continued)
            "CustBookedFlight.BookingData.CultureCode",
            # Interaction history — email outbound
            "IH.Email.Outbound.Pending.pyHistoricalOutcomeCount",    # numeric
            "IH.Email.Outbound.Clicked.pxLastGroupID",
            # Flight data
            "CustBookedFlight.FlightData.DepartureAirport",
            "CustBookedFlight.BookingData.FlightInboundArrival",
            "IH.Email.Outbound.Pending.pxLastGroupID",
            # Interaction history — email inbound (continued)
            "IH.Email.Inbound.Delivered.pxLastGroupID",
            "IH.Email.Inbound.Pending.pxLastGroupID",
            # Flight & passenger (continued)
            "CustBookedFlight.SeatNumber",
            "IH.Email.Inbound.Clicked.pyHistoricalOutcomeCount",    # numeric
            "CustBookedFlight.FlightData.AircraftType",
            "IH.Email.Inbound.Pending.pyHistoricalOutcomeCount",    # numeric
            "IH.Email.Inbound.Clicked.pxLastOutcomeTime.DaysSince",  # numeric
            "IH.Email.Outbound.Clicked.pxLastOutcomeTime.DaysSince", # numeric
            "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince", # numeric
            "IH.Event.Outbound.RealTimeEvent.pxLastOutcomeTime.DaysSince",  # numeric
            "IH.Email.Outbound.Clicked.pyHistoricalOutcomeCount",   # numeric
            "IH.Email.Outbound.Delivered.pxLastOutcomeTime.DaysSince",  # numeric
            "IH.Mobile.Inbound.Impression.pxLastOutcomeTime.DaysSince", # numeric
            "IH.Email.Inbound.Rejected.pyHistoricalOutcomeCount",   # numeric
            # Customer profile
            "Customer.Gender",
            "CustBookedFlight.BrandedFare",
            "IH.Email.Outbound.Delivered.pyHistoricalOutcomeCount", # numeric
            "Customer.MyAccountStatus",
            "IH.Email.Inbound.Rejected.pxLastGroupID",
            "CustBookedFlight.Journey",
            "IH.Email.Inbound.Rejected.pxLastOutcomeTime.DaysSince",  # numeric
            "IH.Web.Inbound.Impression.pxLastOutcomeTime.DaysSince",   # numeric
        ),
        numeric=frozenset({
            "CustBookedFlight.BookingData.DepartureMonth",
            "IH.Push.Outbound.Pending.pyHistoricalOutcomeCount",
            "IH.Email.Inbound.Pending.pxLastOutcomeTime.DaysSince",
            "IH.Email.Outbound.Pending.pyHistoricalOutcomeCount",
            "IH.Email.Inbound.Clicked.pyHistoricalOutcomeCount",
            "IH.Email.Inbound.Pending.pyHistoricalOutcomeCount",
            "IH.Email.Inbound.Clicked.pxLastOutcomeTime.DaysSince",
            "IH.Email.Outbound.Clicked.pxLastOutcomeTime.DaysSince",
            "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince",
            "IH.Event.Outbound.RealTimeEvent.pxLastOutcomeTime.DaysSince",
            "IH.Email.Outbound.Clicked.pyHistoricalOutcomeCount",
            "IH.Email.Outbound.Delivered.pxLastOutcomeTime.DaysSince",
            "IH.Mobile.Inbound.Impression.pxLastOutcomeTime.DaysSince",
            "IH.Email.Inbound.Rejected.pyHistoricalOutcomeCount",
            "IH.Email.Outbound.Delivered.pyHistoricalOutcomeCount",
            "IH.Email.Inbound.Rejected.pxLastOutcomeTime.DaysSince",
            "IH.Web.Inbound.Impression.pxLastOutcomeTime.DaysSince",
        }),
    ),

    # ── Cartrawler: car-rental cross-sell (third-party) ───────────────────────
    "Cartrawler": VariantConfig(
        features=(
            # Scoring parameters
            "param::Param.BundleName",
            # Booking context
            "CustBookedFlight.BookingData.CultureCode",
            "CustBookedFlight.FlightData.AirlineCodeIATA",
            "CustBookedFlight.BookingData.FlightInboundArrival",
            "CustBookedFlight.FlightData.DepartureAirport",
            "CustBookedFlight.BookingData.DurationOfJourney",      # numeric
            "CustBookedFlight.BookingData.FlightInboundDeparture",
            # Interaction history — push
            "IH.Push.Outbound.Pending.pxLastOutcomeTime.DaysSince",  # numeric
            # Flight & passenger
            "CustBookedFlight.Language",
            # Interaction history — email outbound
            "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince",  # numeric
            "IH.Push.Outbound.Pending.pxLastGroupID",
            # Flight data
            "CustBookedFlight.FlightData.AircraftType",
            "IH.Email.Outbound.Clicked.pxLastGroupID",
            "IH.Email.Outbound.Delivered.pyHistoricalOutcomeCount", # numeric
            "IH.Email.Outbound.Delivered.pxLastOutcomeTime.DaysSince",  # numeric
            "CustBookedFlight.FlightData.DestinationAirport",
            "CustBookedFlight.DepartureAirport",
            "CustBookedFlight.IsStaffStandBy",
            # Interaction history — email inbound
            "IH.Email.Inbound.Clicked.pyHistoricalOutcomeCount",    # numeric
            "IH.Email.Inbound.Clicked.pxLastOutcomeTime.DaysSince", # numeric
            # Interaction history — mobile
            "IH.Mobile.Inbound.Impression.pxLastOutcomeTime.DaysSince",  # numeric
            # Interaction history — email outbound (continued)
            "IH.Email.Outbound.Clicked.pyHistoricalOutcomeCount",   # numeric
            # Flight & passenger (continued)
            "CustBookedFlight.SeatNumber",
            # Scoring parameters (continued)
            "param::Param.FlightCost",                              # numeric
            "param::Param.Age",
        ),
        numeric=frozenset({
            "CustBookedFlight.BookingData.DurationOfJourney",
            "IH.Push.Outbound.Pending.pxLastOutcomeTime.DaysSince",
            "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince",
            "IH.Email.Outbound.Delivered.pyHistoricalOutcomeCount",
            "IH.Email.Outbound.Delivered.pxLastOutcomeTime.DaysSince",
            "IH.Email.Inbound.Clicked.pyHistoricalOutcomeCount",
            "IH.Email.Inbound.Clicked.pxLastOutcomeTime.DaysSince",
            "IH.Mobile.Inbound.Impression.pxLastOutcomeTime.DaysSince",
            "IH.Email.Outbound.Clicked.pyHistoricalOutcomeCount",
            "param::Param.FlightCost",
        }),
    ),
}

# ── Primary Pega model IDs per variant (Email/Outbound, highest AUC snapshot) ─
# Used by PegaBinEncoder to load the correct binning from the ADM snapshot file.
PEGA_MODEL_IDS: dict[str, str] = {
    "L5B15":         "340a718b-9899-5637-b3be-1b1e660ef365",  # 24 active predictors, AUC=0.77
    "CLUG":          "80b3b6a6-2527-5330-92b2-0cf46521ff86",  # 21 active predictors, AUC=0.76
    "L5B25":         "f51f1eef-55b3-543b-b755-8e0ea24e7200",  # 22 active predictors, AUC=0.78
    "BookingDotCom": "7e421023-f0fa-5aba-b2cf-68e03ad0d73b",  # 36 active predictors, AUC=0.71
    "Cartrawler":    "a9a0bda9-079d-59a9-9c4e-3352759323bb",  # 25 active predictors, AUC=0.68
}
