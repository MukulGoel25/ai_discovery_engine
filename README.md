# SIP Discovery Engine

AI discovery engine for the question: **why do first-time mutual fund investors stop their SIPs within 12 months, and which problems would move 12-month continuation most?**

It collects public conversations (Play Store, App Store, Reddit, TradingQ&A, YouTube, plus any CSV you add), codes every post into a structured record with a free LLM (Google Gemini by default), verifies the coding, and scores opportunities against the metric tree. A Streamlit app presents everything and lets anyone test the workflow live.

- **Discovery engine (this repo):** hosted on Streamlit Community Cloud.
- **AI-native MVP (separate repo):** https://sip-spend-planner.vercel.app

The app shows real coded data once the pipeline has run. `python pipeline.py demo` switches back to synthetic preview data, marked by a yellow banner.

---

## 1. Set up (Windows, about 5 minutes, ₹0)

```bat
cd sip-discovery-engine
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Everything runs on free tiers. Fill in `.env`:

| Key | Needed for | Where to get it (free, no card) |
|---|---|---|
| `GEMINI_API_KEY` | AI coding, findings, Ask the data | aistudio.google.com → **Get API key** → Create API key |
| `YOUTUBE_API_KEY` | YouTube comments (optional) | console.cloud.google.com → new project → enable **YouTube Data API v3** → Credentials → API key |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reliable Reddit access (optional) | reddit.com/prefs/apps → create app → type **script** |

Other free AI options, set with `LLM_PROVIDER`:
- `groq`: key from console.groq.com. Fast, separate free limits; a good backup when Gemini's daily quota runs out.
- `ollama`: install from ollama.com, run `ollama pull llama3.1`, set `LLM_PROVIDER=ollama` and `EXTRACT_BATCH=1`. Unlimited and offline, but slower and less accurate.

Models are picked automatically from what your key can use (a Flash-Lite model for coding, a Flash model for findings). To pin them, set `EXTRACT_MODEL` / `SYNTH_MODEL`. Pinning also keeps the cache valid if Google adds new models mid-project.

On the free tier Google may use prompts to improve its models. That's fine here: everything sent is public posts, with personal details already removed.

Preview the app with demo data:

```bat
streamlit run app.py
```

## 2. Run the real pipeline

Start small, then run everything:

```bat
python pipeline.py collect
python pipeline.py clean
python pipeline.py extract --limit 100
streamlit run app.py
```

Happy with the quality? Run the rest:

```bat
python pipeline.py extract
python pipeline.py themes
python pipeline.py insights
```

**Free-tier limits.** Free tiers cap requests per minute and per day (see your limits in AI Studio). The engine codes `EXTRACT_BATCH` posts per request (default 8) and waits between requests (`LLM_RPM`, default 8 per minute), so 2,000 posts is about 250 requests. If the daily quota runs out, it stops with "STOPPED: Free daily quota reached". Everything already coded is saved, so run the same command the next day and it continues where it left off. Or switch to `LLM_PROVIDER=groq` for the remainder (the cache is per model, so Groq codes only what's missing in its own run; keep one provider for the final dataset if you can).

| Step | What it does | Output |
|---|---|---|
| `collect` | Pulls reviews, posts and comments. Edit `config/sources.yaml` to change apps, subreddits and search terms. `--only reddit,youtube` runs some collectors. | `data/raw/*.jsonl` |
| `clean` | Removes personal details, short and old posts, duplicates; detects English, Hindi and Hinglish; keeps posts about SIPs, MFs, mandates and autopay; caps the total at `max_items_for_llm` (2,000). | `data/processed/clean.parquet`, `funnel.json` |
| `extract` | The AI codes each post into 25 fields. Replies are forced to JSON and checked against the allowed values; quotes are checked word for word. Cached and resumable. | `data/processed/items.parquet` |
| `themes` | Groups unmet needs bottom-up (TF-IDF + k-means) and names each group. | `clusters.json` |
| `insights` | Findings for each of the 9 research questions from computed numbers only; cited posts are verified. | `insights.json` |
| `propose-taxonomy` | The AI suggests themes from your data (open coding). Review `config/taxonomy_proposed.yaml` and merge what you agree with into `config/taxonomy.yaml`. | proposal file |

**Survey.** In the responses Google Sheet: File → Download → CSV. Save as `data/survey/responses.csv`, or upload it in the app's Survey tab. Contact details are dropped automatically. Open-text answers are also fed into the engine on the next `collect`.

**Other sources.** Put any CSV with a `text` column (optional `date`, `url`, `platform`, `source`) in `data/raw/manual/` for X posts, Quora answers and so on.

## 3. Evals (your AI Evidence Log)

```bat
python pipeline.py golden-sample --n 20
```

1. Open `data/golden/golden_to_label.csv`. Fill the `gold_*` columns **yourself, before looking at any AI output**. Allowed values are in `data/golden/LABEL_GUIDE.md`.
2. Save it as `data/golden/golden_set.csv`.
3. Run:

```bat
python pipeline.py eval
python pipeline.py theme-audit --per-theme 10
REM fill is_correct (y/n) in data/golden/theme_audit.csv, then:
python pipeline.py eval
```

Optional second-model check (e.g. a Flash model vs the Flash-Lite default): `python pipeline.py eval --compare-model <model-name>`

Record the places you rejected or corrected AI output in `data/golden/corrections.csv` (id, field, ai_output, your_correction, why) and rerun `eval`.

Results go to `evals/report.json` and `evals/report.md`, and show in the app's **Quality checks** tab:
- agreement and Cohen's kappa per field
- lenient theme agreement (your theme appears in the AI's top 3)
- the root-cause confusion matrix and every disagreement
- the share of quotes found word for word in the source
- theme precision from the audit
- your corrections log

## 4. Deploy (public link for mentors)

The engine is a Streamlit app, so it is hosted on **Streamlit Community Cloud** (free). Vercel runs static pages and short serverless functions; it can't keep a Streamlit server running, which is why the MVP is on Vercel and the engine is not.

1. Push this folder to a GitHub repo. `.gitignore` excludes `.env`, raw data and the cache. **Do commit** `data/processed/`, `evals/` and `data/golden/` so the app has data.
2. Go to share.streamlit.io, sign in with GitHub, then **Create app** and pick the repo, branch `main` and main file `app.py`.
3. In **Advanced settings**, set Python to the version you use locally and paste `.streamlit/secrets.toml.example` into **Secrets** with your real Gemini key and model name.
4. Deploy, then open the URL in an incognito window to check it works without a login.

`MAX_CALLS_PER_SESSION` (default 20) limits FinBot calls per visitor so mentors testing the link don't use up your free daily quota. Collection and extraction run on your machine, not in the app.

## 5. What's in the app

| Tab | Shows |
|---|---|
| Overview | Where intent leaks across the first year, top opportunities, raw-to-evidence funnel |
| Opportunities | Score per theme (frequency × severity × metric link × addressability), metric-tree branches, sliders to adjust addressability, drill-down with quotes, unmet needs and workarounds |
| Research questions | The 9 questions from the brief, each with charts, findings and quotes |
| Segments | Themes by income, age, influence, source of money, app and language; trend over time |
| Evidence | Every coded post, searchable, with the AI's full record and CSV download |
| FinBot | Research assistant over the coded posts: answers only from cited posts, declines advice requests, says when evidence is thin, removes unverified citations |
| Try the extractor | Paste any post and see how it's coded, including the quote check |
| Survey | Your form responses, people who kept going vs those who didn't |
| Quality checks | Eval results |
| How it works | Method, sources and volumes, taxonomy, the extraction prompt, limitations |

## 6. Troubleshooting

- **Reddit returns 403 or 429.** Add Reddit app credentials to `.env`. Without them Reddit often blocks unauthenticated requests.
- **Play Store returns few reviews for an app.** Check `play_id` in `config/sources.yaml` by opening `https://play.google.com/store/apps/details?id=<play_id>`.
- **"model not found".** Set `EXTRACT_MODEL` / `SYNTH_MODEL` in `.env` to a model listed in AI Studio.
- **"STOPPED: Free daily quota reached".** Normal on free tiers. Rerun tomorrow; it continues from where it stopped.
- **"API key rejected".** Check the key in `.env` has no spaces or quotes.
- **You changed `config/taxonomy.yaml`.** Run `extract` again. The cache is keyed on the taxonomy, so posts are re-coded with the new themes.
- **To go back to demo data:** `python pipeline.py demo`.

## Project layout

```
app.py                 Streamlit app
pipeline.py            CLI for every step
config/                sources.yaml (what to collect), taxonomy.yaml (themes, metric-tree branch, addressability)
engine/collect/        one collector per source
engine/clean.py        cleaning and pre-filter
engine/schema.py       the 25-field record
engine/llm.py          free AI providers (Gemini, Groq, Ollama), JSON checks, rate limits
engine/extract.py      batched AI coding, caching, quote check
engine/analyze.py      counts, opportunity score, segments
engine/themes.py       clusters and taxonomy proposal
engine/insights.py     findings with verified citations
engine/rag.py          Ask the data
engine/evals.py        golden set, kappa, audits
engine/survey.py       survey loader
app_ui/                styles and charts
```

This tool describes investor experiences. It does not give investment advice or recommend funds.
