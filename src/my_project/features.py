"""
Feature configuration for L5B15 Email/Outbound modelling.

PEGA_FEATURES: the variables Pega ADM used for L5B15 scoring.
  - Uncomment each variable Pega actually used.
  - If left empty, notebook 05 falls back to all non-excluded columns.

EXCLUDE_COLS: columns never used as model inputs.
  - Identifiers, PII, partition constants, and Pega output metadata.
"""

TARGET = "propensity"

# ── Columns to always exclude ──────────────────────────────────────────────

_ID_COLS = [
    "pxInteractionID",
    "pxSubjectID",
    "modelExecutionID",
    "TreatmentID",
    "modelVersion",          # model metadata, not a feature
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


# ── Pega ADM feature list ──────────────────────────────────────────────────
# Uncomment the variables Pega used for L5B15 scoring.
# Grouped by namespace for readability.

PEGA_FEATURES = [

    # ── CustBookedFlight: booking context ─────────────────────────────────
    # "CustBookedFlight.BookingData.DaysBetweenBookingAndFlight",
    # "CustBookedFlight.BookingData.FlightCost",
    # "CustBookedFlight.BookingData.JourneyType",
    # "CustBookedFlight.BookingData.IsOneWay",
    # "CustBookedFlight.BookingData.NumberOfAdults",
    # "CustBookedFlight.BookingData.NumberOfChildren",
    # "CustBookedFlight.BookingData.NumberOfBabies",
    # "CustBookedFlight.BookingData.BookingMonth",
    # "CustBookedFlight.BookingData.DepartureMonth",
    # "CustBookedFlight.BookingData.DurationOfJourney",
    # "CustBookedFlight.BookingData.BookingChannel",
    # "CustBookedFlight.BookingData.BookerGender",
    # "CustBookedFlight.BookingData.BookerLanguage",
    # "CustBookedFlight.BookingData.CultureCode",
    # "CustBookedFlight.BookingData.DepartmentCode",
    # "CustBookedFlight.BookingData.CreatedAgentCode",
    # "CustBookedFlight.BookingData.IsASA",

    # ── CustBookedFlight: flight & passenger ──────────────────────────────
    # "CustBookedFlight.DepartureAirport",
    # "CustBookedFlight.ArrivalAirport",
    # "CustBookedFlight.BrandedFare",
    # "CustBookedFlight.ProductClass",
    # "CustBookedFlight.FareClass",
    # "CustBookedFlight.FareBasisCode",
    # "CustBookedFlight.PassengerType",
    # "CustBookedFlight.Gender",
    # "CustBookedFlight.Language",
    # "CustBookedFlight.Loyalty",
    # "CustBookedFlight.Journey",
    # "CustBookedFlight.Role",
    # "CustBookedFlight.FlightNumberOperatorIATA",
    # "CustBookedFlight.CheckinStatus",
    # "CustBookedFlight.SeatNumber",
    # "CustBookedFlight.PassengerID",

    # ── CustBookedFlight: flight status ───────────────────────────────────
    # "CustBookedFlight.FlightData.FlightStatus",
    # "CustBookedFlight.FlightData.AircraftType",
    # "CustBookedFlight.FlightData.ArrivalDelayInMinutes",
    # "CustBookedFlight.FlightData.DepartureDelayInMinutes",
    # "CustBookedFlight.FlightData.IsFlightCancelled",
    # "CustBookedFlight.FlightData.IsPostponedFlight",
    # "CustBookedFlight.FlightData.DelayReason",
    # "CustBookedFlight.FlightData.Departed",
    # "CustBookedFlight.FlightData.Arrived",

    # ── Customer: profile ─────────────────────────────────────────────────
    # "Customer.Gender",
    # "Customer.Language",
    # "Customer.CultureCode",
    # "Customer.MyAccountStatus",
    # "Customer.CustPreference.FavoriteDepartureAirport",
    # "Customer.CustPreference.CalculatedFavoriteAirport",

    # ── Customer: purchase history KPIs ───────────────────────────────────
    # "Customer.CustomerKPIs.CLUGCountTotal",    # past luggage purchases (any)
    # "Customer.CustomerKPIs.L5B15CountTotal",   # past L5B15 (15kg) purchases
    # "Customer.CustomerKPIs.L5B20CountTotal",
    # "Customer.CustomerKPIs.L5B25CountTotal",
    # "Customer.CustomerKPIs.L5B30CountTotal",
    # "Customer.CustomerKPIs.L5B40CountTotal",
    # "Customer.CustomerKPIs.L5B50CountTotal",
    # "Customer.CustomerKPIs.SeatFrontCountTotal",
    # "Customer.CustomerKPIs.SeatRow1CountTotal",
    # "Customer.CustomerKPIs.SeatXLCountTotal",
    # "Customer.CustomerKPIs.SeatNormalCountTotal",
    # "Customer.CustomerKPIs.FASTCountTotal",    # fast-track
    # "Customer.CustomerKPIs.PETCCountTotal",    # pet in cabin
    # "Customer.CustomerKPIs.BIKECountTotal",
    # "Customer.CustomerKPIs.SKISCountTotal",
    # "Customer.CustomerKPIs.GOLFCountTotal",
    # "Customer.CustomerKPIs.SURFCountTotal",
    # "Customer.CustomerKPIs.DIVECountTotal",
    # "Customer.CustomerKPIs.FISHCountTotal",
    # "Customer.CustomerKPIs.HOSTCountTotal",
    # "Customer.CustomerKPIs.AVIHCountTotal",
    # "Customer.CustomerKPIs.DELTCountTotal",

    # ── Interaction History ────────────────────────────────────────────────
    # "IH.Email.Outbound.Pending.pyHistoricalOutcomeCount",
    # "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince",
    # "IH.Email.Outbound.Pending.pxLastGroupID",
    # "IH.Email.Outbound.Delivered.pyHistoricalOutcomeCount",
    # "IH.Email.Outbound.Delivered.pxLastOutcomeTime.DaysSince",
    # "IH.Email.Outbound.Delivered.pxLastGroupID",
    # "IH.Email.Outbound.Clicked.pyHistoricalOutcomeCount",
    # "IH.Email.Outbound.Clicked.pxLastOutcomeTime.DaysSince",
    # "IH.Email.Outbound.Clicked.pxLastGroupID",
    # "IH.Push.Outbound.Pending.pyHistoricalOutcomeCount",
    # "IH.Push.Outbound.Pending.pxLastOutcomeTime.DaysSince",
    # "IH.Push.Outbound.Pending.pxLastGroupID",
    # "IH.Mobile.Inbound.Impression.pyHistoricalOutcomeCount",
    # "IH.Mobile.Inbound.Impression.pxLastOutcomeTime.DaysSince",
    # "IH.Event.Outbound.RealTimeEvent.pyHistoricalOutcomeCount",
    # "IH.Event.Outbound.RealTimeEvent.pxLastOutcomeTime.DaysSince",

    # ── Model parameters passed by Pega ───────────────────────────────────
    # "param::Param.Age",
    # "param::Param.BundleName",
    # "param::Param.FlightCost",
    # "param::Param.JourneyType",
    # "param::Param.MonthNumber",
]
