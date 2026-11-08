# How to label the golden set

Label each row BEFORE looking at any AI output.

- gold_is_relevant: yes / no (is it about a person's own SIP / mutual-fund experience?)
- gold_event: one of started, continuing, considering_stop, paused, reduced, skipped, stopped, failed_instalment, switched_fund, switched_platform, restarted, none
- gold_primary_theme: one of failed_payment_mandate, debit_date_mismatch, expectation_mismatch, confusing_metrics, check_anxiety, no_goal_anchor, cash_flow_life_event, irregular_income, pause_never_resumed, social_influence_churn, market_fall_trigger, fund_choice_doubt, app_trust_ops, clutter_cross_sell, positive_habit, none
- gold_journey_stage: one of before_start, start, first_debit, months_1_3, months_4_6, months_7_12, after_12_months, unclear
- gold_first_time: one of yes, no, unclear
- gold_notes: anything ambiguous (sarcasm, Hinglish, mixed reasons)

Save the labelled file as data/golden/golden_set.csv
