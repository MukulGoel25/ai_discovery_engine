"""Synthetic DEMO data so the app can be previewed before the real pipeline runs.

Everything here is invented. The app shows a banner whenever data/processed/DEMO_DATA exists.
Running the real pipeline (`python pipeline.py all`) removes the flag and overwrites the files.
"""
from __future__ import annotations

import json
import random

import pandas as pd

from .settings import CLUSTERS_FILE, DEMO_FLAG, EVALS, FUNNEL_FILE, INSIGHTS_FILE, ITEMS_FILE

T = {
    "failed_payment_mandate": [
        "My SIP failed twice this month and the app only says failed, no reason. Bank also cut a bounce charge.",
        "Mandate pending for 10 days, first instalment never went. Kab tak wait karu?",
        "Money got debited but units not allotted, SIP shows processing since a week.",
        "Changed my salary account and all SIPs started failing, nobody told me the mandate was linked to old bank.",
    ],
    "debit_date_mismatch": [
        "SIP date is 2nd and salary comes on 30th, rent goes on 1st so balance is short every time.",
        "I picked the date I signed up on, didn't think about salary date. Two failures already.",
        "Salary late this month due to holiday and SIP bounced, again a charge.",
    ],
    "expectation_mismatch": [
        "Six months of SIP and my return is negative. Everyone said 12-15 percent, what is the point?",
        "Started after watching a video showing crores in 20 years, my SIP is barely up 300 rupees.",
        "Thought SIP never goes in loss if you do it monthly. Now it is red, feeling cheated.",
    ],
    "confusing_metrics": [
        "Fund shows 22% 3 year return but my XIRR is -8%. Is the app showing wrong numbers?",
        "Can someone explain XIRR? My absolute return is 2% but XIRR says 9%, which one is true?",
        "Why does my value go down when I keep investing every month? Confused about NAV.",
    ],
    "check_anxiety": [
        "I check the app 4-5 times a day and every red day ruins my mood. Thinking to stop SIP for peace.",
        "Daily change notification in red makes me anxious, I paused to take a break from stress.",
    ],
    "no_goal_anchor": [
        "Started SIP because my whole team was doing it, never had any goal. Now it feels optional.",
        "Office session pe SIP start kar diya tha, ab samajh nahi aata kis liye kar raha hu.",
    ],
    "cash_flow_life_event": [
        "Paused my SIP for sister's wedding expenses, will restart later.",
        "Lost my job last month so stopped all SIPs, markets were also down.",
        "Had to pay rent deposit for new flat so stopped the SIP for now.",
        "Ghar ka kharcha zyada tha is mahine, SIP band kar diya.",
    ],
    "irregular_income": [
        "Freelancer here, some months 20k some months 80k. Fixed SIP date doesn't work for me.",
        "Commission based job, I keep pausing and forgetting to resume.",
    ],
    "pause_never_resumed": [
        "Paused my SIP in March thinking I will restart in 2 months, it's October and still paused.",
        "I stopped thinking I'll restart when markets get better. Never did.",
    ],
    "social_influence_churn": [
        "WhatsApp group said small caps will crash so I stopped my SIP to wait.",
        "A YouTuber said to switch funds, I stopped the old SIP and never started the new one.",
    ],
    "market_fall_trigger": [
        "Market is falling every day, should I stop my SIP and restart later?",
        "Portfolio down 10% because of the crash, stopping SIP till things settle.",
    ],
    "fund_choice_doubt": [
        "Not sure if I picked the right fund, everyone recommends something different.",
    ],
    "app_trust_ops": [
        "Support not responding for 5 days about my SIP order, very poor service.",
        "App shows SIP active but no instalment happened this month. Can't trust it.",
    ],
    "clutter_cross_sell": [
        "Too many notifications about US stocks and IPOs, I just want to see my SIP.",
    ],
    "positive_habit": [
        "Two years of SIP now, I don't check daily. Set it after salary date and forgot about it.",
        "Saw my SIP go red in the first year but continued, it recovered later. Have a goal for house down payment.",
        "My SIP is for my daughter's education, I don't look at daily numbers at all.",
    ],
}
WEIGHTS = {"failed_payment_mandate": 13, "debit_date_mismatch": 8, "expectation_mismatch": 12, "confusing_metrics": 10,
           "check_anxiety": 6, "no_goal_anchor": 5, "cash_flow_life_event": 11, "irregular_income": 4,
           "pause_never_resumed": 6, "social_influence_churn": 4, "market_fall_trigger": 9, "fund_choice_doubt": 5,
           "app_trust_ops": 7, "clutter_cross_sell": 2, "positive_habit": 10}
EVENT_FOR = {"failed_payment_mandate": ["failed_instalment"] * 3 + ["paused", "stopped"],
             "debit_date_mismatch": ["failed_instalment", "failed_instalment", "paused"],
             "expectation_mismatch": ["considering_stop", "stopped", "paused"],
             "confusing_metrics": ["continuing", "considering_stop", "none"],
             "check_anxiety": ["paused", "considering_stop"], "no_goal_anchor": ["paused", "stopped", "continuing"],
             "cash_flow_life_event": ["paused", "stopped", "reduced"], "irregular_income": ["paused", "skipped"],
             "pause_never_resumed": ["paused", "stopped"], "social_influence_churn": ["stopped", "switched_fund"],
             "market_fall_trigger": ["considering_stop", "stopped"], "fund_choice_doubt": ["switched_fund", "none"],
             "app_trust_ops": ["failed_instalment", "switched_platform"], "clutter_cross_sell": ["none", "continuing"],
             "positive_habit": ["continuing", "continuing", "restarted"]}
STAGE_FOR = {"failed_payment_mandate": ["first_debit", "months_1_3", "months_4_6"],
             "debit_date_mismatch": ["first_debit", "months_1_3"],
             "expectation_mismatch": ["months_1_3", "months_4_6", "months_7_12"],
             "confusing_metrics": ["months_1_3", "months_4_6"], "check_anxiety": ["months_1_3", "months_4_6"],
             "cash_flow_life_event": ["months_4_6", "months_7_12"], "pause_never_resumed": ["months_4_6", "months_7_12"],
             "market_fall_trigger": ["months_4_6", "months_7_12"]}
SOURCES = [("playstore", 34), ("reddit", 30), ("youtube", 18), ("appstore", 8), ("tradingqna", 7), ("survey", 3)]
CONF = {"confusing_metrics": ["xirr", "fund_vs_own_return", "value_down_while_investing"],
        "failed_payment_mandate": ["failure_reason", "mandate_process"], "expectation_mismatch": ["value_down_while_investing"]}


def _pick(rng, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def build(n: int = 650, seed: int = 5) -> None:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        th = _pick(rng, WEIGHTS)
        text = rng.choice(T[th])
        src = _pick(rng, dict(SOURCES))
        relevant = rng.random() > 0.12
        sec = rng.sample([t for t in WEIGHTS if t not in (th, "positive_habit")], k=rng.choice([0, 0, 1])) if th != "positive_habit" else []
        if th == "cash_flow_life_event" and "markets" in text:
            sec = ["market_fall_trigger"]
        words = text.split()
        quote = " ".join(words[: min(len(words), rng.randint(8, 16))])
        month = f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}"
        rows.append({
            "id": f"DEMO-{i:04d}", "source": src,
            "platform": rng.choice(["INDmoney", "Groww", "Zerodha Kite"]) if src in ("playstore", "appstore") else "generic",
            "date": month, "url": "", "title": "", "context": "DEMO", "text": text, "full_text": text,
            "rating": rng.randint(1, 5) if src in ("playstore", "appstore") else None, "likes": rng.randint(0, 40),
            "lang": "hinglish" if any(w in text.lower() for w in ("kar", "hu", "ka ", "kab")) else "en",
            "kw_hit": True, "is_relevant": relevant, "asset_type": "mutual_fund_sip",
            "x_platform": rng.choice(["INDmoney", "Groww", "Zerodha", "unknown"]),
            "first_time_investor": rng.choice(["yes", "yes", "unclear", "no"]),
            "journey_stage": rng.choice(STAGE_FOR.get(th, ["months_1_3", "months_4_6", "months_7_12", "unclear"])),
            "event": rng.choice(EVENT_FOR[th]), "themes": [th] + sec, "primary_theme": th,
            "continuation_drivers": (rng.sample(["goal_anchor", "automation_set_and_forget", "understood_volatility",
                                                 "saw_recovery_after_dip"], k=2) if th == "positive_habit"
                                     else rng.sample(["small_affordable_amount", "trusted_person_to_ask"], k=rng.choice([0, 0, 1]))),
            "why_started": rng.choice(["long_term_wealth", "peers_colleagues", "finfluencer_youtube", "specific_goal",
                                       "tax_saving", "unclear", "unclear"]),
            "expectation": rng.choice(["specific_high_return", "never_loss", "moderate_return", "unclear", "unclear"]),
            "checks": rng.sample(["current_vs_invested", "xirr_or_return_pct", "one_day_change", "fund_past_returns"], k=rng.choice([0, 1, 2])),
            "check_frequency": rng.choice(["daily", "multiple_daily", "weekly", "monthly", "unclear", "unclear"]),
            "confusions": CONF.get(th, [])[: rng.randint(0, 2)],
            "outside_app_actions": rng.sample(["youtube", "asked_friend_family", "reddit_forum", "google_search",
                                               "whatsapp_telegram", "app_support"], k=rng.choice([0, 1, 1, 2])),
            "income": rng.choice(["salaried", "salaried", "salaried", "variable_freelance_business", "student", "unclear"])
            if th != "irregular_income" else "variable_freelance_business",
            "age_band": rng.choice(["18_22", "23_28", "23_28", "29_35", "36_plus", "unclear"]),
            "money_source": rng.choice(["salary", "salary", "parents_pocket_money", "bonus_windfall", "unclear"]),
            "influencer": rng.choice(["self", "friend_family", "finfluencer", "employer", "unclear"]),
            "emotion": rng.choice(["frustrated", "anxious", "confused", "disappointed", "neutral"]) if th != "positive_habit" else "confident",
            "severity": rng.choice([2, 3, 3, 4, 4, 5]) if th != "positive_habit" else 1,
            "unmet_need": {"failed_payment_mandate": "Clear reason when an instalment fails and an easy way to fix it",
                           "debit_date_mismatch": "A SIP date that matches when salary arrives",
                           "expectation_mismatch": "Realistic picture of what the first year of a SIP looks like",
                           "confusing_metrics": "Plain explanation of what my own return numbers mean",
                           "check_anxiety": "A calmer way to see progress without daily red numbers",
                           "no_goal_anchor": "A personal reason to keep the SIP going",
                           "cash_flow_life_event": "Option to reduce or skip without stopping completely",
                           "irregular_income": "SIP that adjusts to irregular income",
                           "pause_never_resumed": "A reminder to restart after a pause",
                           "social_influence_churn": "A neutral place to check claims heard from others",
                           "market_fall_trigger": "Context on whether a fall is normal",
                           "fund_choice_doubt": "Confidence in the fund already chosen",
                           "app_trust_ops": "Clear order status and responsive support",
                           "clutter_cross_sell": "Fewer unrelated product pushes",
                           "positive_habit": ""}[th],
            "workaround": rng.choice(["", "Keeps extra balance before debit date", "Asked a friend", "Set a phone reminder"]),
            "evidence_quote": quote, "confidence": round(rng.uniform(0.6, 0.95), 2), "quote_grounded": True,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(ITEMS_FILE, index=False)
    raw = {"playstore": 9100, "appstore": 1250, "reddit": 2600, "youtube": 3400, "tradingqna": 420, "survey": 10}
    FUNNEL_FILE.write_text(json.dumps({"funnel": {"raw": 16780, "after_length_filter": 14020, "after_date_filter": 12910,
                                                  "after_dedupe": 11870, "after_keyword_filter": 2410, "sent_to_llm": n},
                                       "raw_by_source": raw, "by_source": df["source"].value_counts().to_dict()}), encoding="utf-8")
    rel = df[df["is_relevant"]]
    cl = []
    for i, (th, g) in enumerate(rel.groupby("primary_theme")):
        if th == "positive_habit":
            continue
        cl.append({"cluster": i, "name": g["unmet_need"].iloc[0], "description": "DEMO cluster", "maps_to_theme": th,
                   "items": len(g), "share": len(g) / len(rel), "top_terms": [], "examples": g["unmet_need"].head(3).tolist(),
                   "ids": g["id"].tolist()})
    CLUSTERS_FILE.write_text(json.dumps({"k": len(cl), "n": len(rel), "clusters": sorted(cl, key=lambda c: -c["items"])}), encoding="utf-8")
    INSIGHTS_FILE.write_text(json.dumps({"_meta": {"demo": True}}), encoding="utf-8")
    (EVALS / "report.json").write_text(json.dumps({
        "demo": True, "generated": "DEMO", "model": "demo", "golden_n": 20,
        "fields": {"is_relevant": {"n": 20, "agreement": 0.95, "kappa": 0.86},
                   "event": {"n": 18, "agreement": 0.83, "kappa": 0.78},
                   "primary_theme": {"n": 18, "agreement": 0.72, "kappa": 0.69, "lenient_agreement": 0.89,
                                     "confusion": {}},
                   "journey_stage": {"n": 18, "agreement": 0.67, "kappa": 0.58},
                   "first_time_investor": {"n": 18, "agreement": 0.78, "kappa": 0.55}},
        "disagreements": [{"id": "DEMO-0007", "field": "primary_theme", "gold": "cash_flow_life_event",
                           "ai": "market_fall_trigger", "text": "Lost my job last month so stopped all SIPs, markets were also down.",
                           "note": "DEMO example"}],
        "grounding": {"all": 0.97, "relevant": 0.98, "n": n}}, indent=2), encoding="utf-8")
    DEMO_FLAG.write_text("synthetic demo data", encoding="utf-8")
    print(f"  wrote {n} synthetic DEMO items (banner will show in the app)")
