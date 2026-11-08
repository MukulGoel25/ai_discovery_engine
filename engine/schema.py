"""The structured record the LLM fills for every post/review/comment.

Turning messy text into these fields is what lets us count, compare and segment
instead of 'summarising reviews'.
"""
from __future__ import annotations

from .settings import taxonomy_cfg, theme_ids

PLATFORMS = ["INDmoney", "Groww", "Zerodha", "Paytm Money", "ET Money", "AMC app", "other", "unknown"]
ASSET_TYPES = ["mutual_fund_sip", "stock_sip", "other_investment", "unclear"]
FIRST_TIME = ["yes", "no", "unclear"]
STAGES = ["before_start", "start", "first_debit", "months_1_3", "months_4_6",
          "months_7_12", "after_12_months", "unclear"]
EVENTS = ["started", "continuing", "considering_stop", "paused", "reduced", "skipped",
          "stopped", "failed_instalment", "switched_fund", "switched_platform", "restarted", "none"]
DROP_EVENTS = {"paused", "stopped", "failed_instalment", "considering_stop", "reduced", "skipped", "switched_platform"}
WHY_STARTED = ["long_term_wealth", "specific_goal", "tax_saving", "peers_colleagues", "finfluencer_youtube",
               "family", "employer_session", "app_ad_offer", "unclear"]
EXPECTATION = ["specific_high_return", "moderate_return", "never_loss", "no_expectation", "unclear"]
CHECKS = ["current_vs_invested", "pnl_rupees", "xirr_or_return_pct", "one_day_change",
          "fund_past_returns", "next_instalment", "nothing", "unclear"]
CHECK_FREQ = ["multiple_daily", "daily", "weekly", "monthly", "rarely", "unclear"]
CONFUSIONS = ["xirr", "fund_vs_own_return", "value_down_while_investing", "nav_allotment_timing",
              "failure_reason", "pause_vs_stop", "tax_exit_load", "mandate_process", "none"]
OUTSIDE = ["asked_friend_family", "youtube", "reddit_forum", "whatsapp_telegram", "google_search",
           "ai_chatbot", "app_support", "adviser", "none"]
INCOME = ["salaried", "variable_freelance_business", "student", "unclear"]
AGE = ["18_22", "23_28", "29_35", "36_plus", "unclear"]
MONEY_SOURCE = ["salary", "parents_pocket_money", "bonus_windfall", "side_income", "unclear"]
INFLUENCER = ["self", "friend_family", "finfluencer", "employer", "app_ad", "unclear"]
EMOTION = ["anxious", "frustrated", "confused", "regretful", "disappointed", "confident", "neutral"]


def drivers() -> list[str]:
    return taxonomy_cfg().get("continuation_drivers", []) + ["none"]


def tool_schema() -> dict:
    """JSON schema passed to the LLM as a tool, which forces structured output."""
    themes = theme_ids()
    arr = lambda vals: {"type": "array", "items": {"type": "string", "enum": vals}}  # noqa: E731
    return {
        "name": "record_extraction",
        "description": "Record the structured coding of one public post about SIP / mutual-fund investing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "is_relevant": {"type": "boolean",
                                "description": "True only if the text describes a person's OWN (or a close person's) SIP/mutual-fund investing experience, decision or problem."},
                "asset_type": {"type": "string", "enum": ASSET_TYPES},
                "platform": {"type": "string", "enum": PLATFORMS},
                "first_time_investor": {"type": "string", "enum": FIRST_TIME},
                "journey_stage": {"type": "string", "enum": STAGES},
                "event": {"type": "string", "enum": EVENTS},
                "themes": {**arr(themes), "maxItems": 3},
                "primary_theme": {"type": "string", "enum": themes + ["none"]},
                "continuation_drivers": arr(drivers()),
                "why_started": {"type": "string", "enum": WHY_STARTED},
                "expectation": {"type": "string", "enum": EXPECTATION},
                "checks": arr(CHECKS),
                "check_frequency": {"type": "string", "enum": CHECK_FREQ},
                "confusions": arr(CONFUSIONS),
                "outside_app_actions": arr(OUTSIDE),
                "income": {"type": "string", "enum": INCOME},
                "age_band": {"type": "string", "enum": AGE},
                "money_source": {"type": "string", "enum": MONEY_SOURCE},
                "influencer": {"type": "string", "enum": INFLUENCER},
                "emotion": {"type": "string", "enum": EMOTION},
                "severity": {"type": "integer", "minimum": 1, "maximum": 5,
                             "description": "1 = mild annoyance, 5 = stopped investing / lost money / gave up."},
                "unmet_need": {"type": "string", "description": "One sentence, in plain words, of what the person needed but didn't get. Empty if none."},
                "workaround": {"type": "string", "description": "What they did instead (e.g., kept extra balance, asked a friend). Empty if none."},
                "evidence_quote": {"type": "string", "description": "Up to 25 words copied EXACTLY from the text that best supports primary_theme."},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["is_relevant", "asset_type", "platform", "first_time_investor", "journey_stage",
                         "event", "themes", "primary_theme", "continuation_drivers", "why_started",
                         "expectation", "checks", "check_frequency", "confusions", "outside_app_actions",
                         "income", "age_band", "money_source", "influencer", "emotion", "severity",
                         "unmet_need", "workaround", "evidence_quote", "confidence"],
        },
    }


EMPTY_EXTRACTION = {
    "is_relevant": False, "asset_type": "unclear", "platform": "unknown", "first_time_investor": "unclear",
    "journey_stage": "unclear", "event": "none", "themes": [], "primary_theme": "none",
    "continuation_drivers": [], "why_started": "unclear", "expectation": "unclear", "checks": [],
    "check_frequency": "unclear", "confusions": [], "outside_app_actions": [], "income": "unclear",
    "age_band": "unclear", "money_source": "unclear", "influencer": "unclear", "emotion": "neutral",
    "severity": 1, "unmet_need": "", "workaround": "", "evidence_quote": "", "confidence": 0.0,
}
