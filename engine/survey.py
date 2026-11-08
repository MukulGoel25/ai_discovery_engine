"""Load the Google Form responses (Sheet -> File -> Download -> CSV) and make them analysable.

Google Forms joins checkbox answers with ', ' and several options contain commas, so multi-select
cells are parsed by matching the known option texts; anything left over is counted as 'Other'.
Contact details (name/phone/email) are dropped before anything is displayed.
"""
from __future__ import annotations

import re

import pandas as pd

from .settings import SURVEY

MULTI = {
    "most influenced": ['Friend or colleague', 'Parent / sibling / relative', 'YouTube / Instagram / finfluencer',
                        'Reddit / online forum', 'News / article', 'App advertisement or notification',
                        'Employer or HR session', 'Financial adviser / distributor', 'My own research'],
    "look at first": ['Current value vs amount invested', 'Profit/loss in ₹', 'Return % / XIRR', "Today's change (1-day)",
                      "Fund's 1Y/3Y/5Y returns", 'Next instalment date', 'Nothing specific'],
    "found confusing": ['What XIRR means', "Why my return differs from the fund's return",
                        'Why value went down even though I keep investing', 'When units get allotted / NAV date',
                        'Why an instalment failed', 'Difference between pause and stop', 'Tax / exit load',
                        'Nothing was confusing'],
    "issue(s) did you face": ['Mandate / autopay registration or approval failed', 'Payment or debit failed',
                              'Not enough balance on the debit date',
                              'Money was debited but units were not bought / order failed',
                              'Order stayed pending for many days', 'Order was rejected or reversed', 'Refund was delayed',
                              'Changed bank account and the SIP stopped working',
                              'Bank charged a bounce / failed-debit fee', "Couldn't tell whether the SIP was still active",
                              "The next instalment didn't happen as expected", 'Not sure'],
    "what was the reason": ['Not enough balance on that day', 'Mandate / autopay issue', 'Changed bank account',
                            'Bank/technical error', "I don't know the reason", 'Not applicable'],
    "what did you do after the issue": ['Waited for the status to change', 'Retried / paid the instalment manually',
                                        'Checked bank SMS/statement', 'Read the app’s help pages', 'Contacted app support',
                                        'Asked a friend or family member', 'Searched online / YouTube',
                                        'Changed the SIP date or amount', 'Paused the SIP', 'Stopped the SIP',
                                        'Started another SIP', 'Moved (or considered moving) to another app', 'Did nothing'],
    "would have helped": ['What exactly failed and why', 'Whether money had left my account', 'Whether my SIP was still active',
                          'When the next instalment would happen', 'Whether to wait or retry', 'What to do next',
                          'Refund status', 'How to reach the right support', 'None of these'],
    "ever done any of these": ['Paused it', 'Skipped an instalment', 'Reduced the amount', 'Stopped / cancelled it',
                               "Seriously considered stopping but didn't", 'Switched to a different fund',
                               'Moved my SIP to another app', 'None of these'],
    "going on at that time": ['A big expense (rent deposit, wedding, medical, travel, gadget)',
                              'Income dropped / job change / delay', 'Portfolio was showing a loss',
                              'Returns felt too small to matter', 'News or people said markets would fall',
                              'A friend/creator suggested a different fund or investment',
                              'Instalment failures / app problems',
                              'I needed money for something else and SIP felt optional',
                              'Lost interest / forgot about it', 'Not applicable'],
    "outside the app": ['Ask a friend / colleague', 'Ask family', 'Watch YouTube videos', 'Search Google',
                        'Post or read on Reddit / forums', 'Ask in WhatsApp/Telegram groups', 'Ask an AI chatbot',
                        'Call or chat with app support', 'Talk to an adviser', 'Nothing'],
    "more likely to continue": ['Understanding what my numbers mean', 'A debit date that matches when I get paid',
                                'Being able to lower the amount easily in tight months', 'Seeing progress toward my own goal',
                                'Fewer, calmer updates instead of daily ups and downs',
                                'A reminder before the debit to keep balance', 'Clear explanation when an instalment fails',
                                'Knowing what a normal first year looks like', 'Someone neutral to ask questions (not sales)',
                                "Nothing – I'll continue anyway"],
}
PRIVATE = ("phone", "email", "name and", "your name", "contact")
FREE_TEXT = ("anything else",)


def find_col(df: pd.DataFrame, *keys: str) -> str | None:
    for c in df.columns:
        lc = c.lower()
        if all(k.lower() in lc for k in keys):
            return c
    return None


def load(path_or_buffer=None) -> pd.DataFrame | None:
    src = path_or_buffer if path_or_buffer is not None else SURVEY / "responses.csv"
    if hasattr(src, "exists") and not src.exists():
        return None
    df = pd.read_csv(src)
    df = df[[c for c in df.columns if not any(p in c.lower() for p in PRIVATE)]]
    return df


def multi_key(col: str) -> str | None:
    lc = col.lower()
    for k in MULTI:
        if k in lc:
            return k
    return None


def split_multi(cell, options: list[str]) -> list[str]:
    if pd.isna(cell) or not str(cell).strip():
        return []
    s = str(cell)
    found = []
    for o in sorted(options, key=len, reverse=True):
        if o in s:
            found.append(o)
            s = s.replace(o, "")
    rest = re.sub(r"^[\s,]+|[\s,]+$", "", re.sub(r"(,\s*)+", ", ", s))
    if rest:
        found.append("Other")
    return found


def question_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.match(r"^\s*\d+\.", c)]


def counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    k = multi_key(col)
    if k:
        s = df[col].map(lambda v: split_multi(v, MULTI[k])).explode().dropna()
    else:
        s = df[col].dropna().astype(str)
    vc = s.value_counts()
    return pd.DataFrame({"answer": vc.index, "n": vc.values, "share": vc.values / max(df[col].notna().sum(), 1)})


def flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    q_status = find_col(df, "ever started a sip")
    q_done = find_col(df, "ever done any of these")
    drop = pd.Series(False, index=df.index)
    if q_status:
        drop |= df[q_status].astype(str).str.contains("paused|stopped|never went through", case=False, na=False)
    if q_done:
        drop |= df[q_done].astype(str).str.contains("Paused it|Stopped|Skipped|Reduced", case=False, na=False)
    out["group"] = drop.map({True: "Paused / stopped / reduced", False: "Kept going"})
    q_issue = find_col(df, "payment or sip issue") or find_col(df, "instalment ever failed")
    out["had_issue"] = df[q_issue].astype(str).str.startswith("Yes") if q_issue else False
    q_first = find_col(df, "any mutual fund investment") or find_col(df, "ever invested in mutual funds")
    out["first_timer"] = df[q_first].astype(str).str.startswith("No") if q_first else True
    return out


CROSSTABS = [
    ("How they chose the debit date", ("debit date",)),
    ("Goal linked to the SIP", ("specific goal", "name it")),
    ("Expected yearly return", ("yearly return",)),
    ("How often they check", ("how often",)),
    ("Income stability", ("how stable",)),
    ("Main reason for starting", ("main reason you started",)),
    ("Emergency buffer", ("emergencies",)),
    ("Share of savings", ("share of your monthly savings",)),
]


def crosstab(df: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame | None:
    col = find_col(df, *keys)
    if not col:
        return None
    f = flags(df)
    t = pd.crosstab(df[col].fillna("No answer"), f["group"], normalize="columns")
    return t.reset_index().melt(id_vars=col, var_name="group", value_name="share").rename(columns={col: "answer"})
