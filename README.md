# Omnichannel Chatbot Quality Gate

**A release gate for multi-turn chatbots. It tests the same conversations through the REST API and through the live web chat widget, scores the answers with an LLM judge, and blocks the merge when quality drops.**

It is built to test an *existing, external* chatbot: the target is configured through `.env` (URL, API field names, widget selectors), not coded in. A small mock chatbot is bundled so anyone can clone the repository and watch the whole pipeline run with one command.

`Python` · `Playwright` · `DeepEval` · `httpx` · `Pydantic v2` · `pytest` · `FastAPI` (mock target) · `Docker` · `GitHub Actions`

---

## The problem this solves

A chatbot release can go wrong in three different ways, and each needs a different kind of test:

| What breaks | Example | What catches it |
| --- | --- | --- |
| **The channel** | The widget stops rendering replies; the API returns 503; one customer's order number shows up in another session. | Deterministic API and UI checks. Exact, fast, free. |
| **Hard rules** | The bot reveals its system prompt; it forgets the order number the customer gave three turns ago; it does not hand over when asked for a human. | Deterministic checks on the transcript: canary tokens, required phrases, the escalation signal. |
| **Answer quality** | The bot adds a plausible but invented claim; the answer drifts off topic; the tone turns curt. | An LLM judge (DeepEval) with numeric thresholds. |

Running only the LLM judge is slow, costs tokens, and gives a fuzzy answer to questions that have an exact one. Running only deterministic checks misses hallucinations entirely. This framework runs both, **cheapest first**.

## Architecture

```
 ┌──────────────────────────┐
 │      Golden dataset      │  datasets/golden/*.json, validated by Pydantic
 │  RAG questions           │  + reviewed synthetic goldens (DeepEval Synthesizer)
 │  multi-turn scenarios    │
 │  guardrail probes        │
 └────────────┬─────────────┘
              │ scripted user messages
              ▼
 ┌──────────────────────────┐        ┌───────────────────────────────────────────┐
 │    ConversationRunner    │──────► │ AdapterFactory.create(channel)            │
 │  plays a conversation,   │        │                                           │
 │  records a Transcript,   │        │   BaseChatbotAdapter      (Strategy)      │
 │  writes a failure trace  │        │    ├── RestApiAdapter      httpx, retries,│
 │  when the channel breaks │        │    │                       latency        │
 └────────────┬─────────────┘        │    └── PlaywrightUiAdapter live widget,   │
              │                      │                            screenshot +   │
              │                      │                            trace on fail  │
              │                      └───────────────────┬───────────────────────┘
              │                                          │ HTTP / browser
              │                                          ▼
              │                      ┌───────────────────────────────────────────┐
              │                      │ Target chatbot (any URL, set in .env)     │
              │                      │   REST API            Web chat widget     │
              │                      │ default: bundled mock "Nimbus Telecom" bot│
              │                      └───────────────────────────────────────────┘
              │ Transcript (same shape for every channel)
              ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                              CI/CD RELEASE GATE                                 │
 │                                                                                 │
 │  Stage 1  API smoke   deterministic   contract, session isolation, retention,   │
 │           (no LLM)                    escalation flag, canary leaks, p95 latency│
 │              │ pass                                                             │
 │              ▼                                                                  │
 │  Stage 2  UI smoke    deterministic   widget opens, Enter sends, a11y labels,   │
 │           (no LLM)                    HTML-injection safety, handover banner,   │
 │              │ pass                   widget/API parity, no console errors      │
 │              ▼                                                                  │
 │  Stage 3  DeepEval judge, once per channel (api, ui)                            │
 │           RAG         Faithfulness ≥ 0.85 · Answer Relevancy ≥ 0.85 ·           │
 │                       Contextual Precision                                      │
 │           Multi-turn  ConversationalTestCase: Knowledge Retention ·             │
 │                       Conversation Completeness                                 │
 │           Guardrails  G-Eval Guardrail Compliance ≥ 0.85 · No Fabrication ≥ 0.85│
 │                       · Toxicity ≤ 0.15                                         │
 │           Custom      G-Eval Brand Tone · G-Eval Escalation Accuracy            │
 │              │                                                                  │
 │              ▼                                                                  │
 │        any failure  ──►  job fails  ──►  merge blocked                          │
 └─────────────────────────────────────────────────────────────────────────────────┘
```

## Run it

### One command

```bash
git clone <your-repo-url> && cd omnichannel-chatbot-quality-gate
cp .env.example .env          # optional: add OPENAI_API_KEY to enable stage 3
make demo
```

`make demo` builds one Docker image, starts the mock chatbot, and runs the gate against it: API smoke, then UI smoke in headless Chromium, then DeepEval. Without a judge API key, stages 1 and 2 still run and stage 3 is skipped with a message. The widget is at `http://localhost:8080` while it runs.

### Watch the gate catch a regression

The mock bot can be started with a known fault:

```bash
make demo-fault FAULT=leak          # obeys prompt injection and reveals its system prompt
make demo-fault FAULT=forget        # loses conversation memory
make demo-fault FAULT=hallucinate   # adds an invented claim to every knowledge answer
```

| Fault | Caught at | By |
| --- | --- | --- |
| `leak` | Stage 1, before any tokens are spent | canary-token check on guardrail goldens |
| `forget` | Stage 1, before any tokens are spent | per-turn recall checks on retention goldens |
| `hallucinate` | Stage 3 | the LLM judge (Faithfulness); string checks cannot see it |

The third row is the reason the judge exists. The unit suite proves that claim: `test_hallucinate_fault_is_invisible_to_deterministic_checks`.

### Without Docker

```bash
python -m venv .venv && source .venv/bin/activate
make install        # framework + mock bot + Chromium
make test           # unit tests: no server, no browser, no keys
make serve          # terminal 1: mock chatbot on :8080
make smoke-api      # terminal 2: stage 1
make smoke-ui       #             stage 2
make eval-api       #             stage 3 through the REST adapter
make eval-ui        #             stage 3 through the Playwright adapter
```

## Why smoke gates run before the LLM judge

Stage 3 runs **54 LLM-judged test cases per channel** on the bundled dataset, and each metric makes several model calls. Stages 1 and 2 make none. Ordering them this way has three effects:

- **Broken builds fail in seconds, for free.** If the widget is down or the API contract changed, there is no point paying a judge to read error messages.
- **Exact questions get exact answers.** "Did the system prompt leak?" and "Is this the order number the customer gave?" are string comparisons. A judge would be slower, costlier and less reliable at them.
- **Judge failures mean something.** By the time stage 3 runs, the channel works and the hard rules hold, so a low score points at answer quality and nothing else.

Stage 3 then does what only a judge can: catch a hallucination regression before a model or prompt change is released. Faithfulness compares every claim in the answer with the knowledge chunks; one invented sentence fails the build.

## The evaluation suite

| File | Metric | Question | Threshold |
| --- | --- | --- | --- |
| `evals/test_rag_quality.py` | `FaithfulnessMetric` | Is every claim supported by the knowledge chunks? | **≥ 0.85** |
| | `AnswerRelevancyMetric` | Does the answer address the question, without padding? | **≥ 0.85** |
| | `ContextualPrecisionMetric` | Is the relevant chunk ranked first? | ≥ 0.70 |
| `evals/test_multi_turn.py` | `KnowledgeRetentionMetric` | Does the bot keep what the user said earlier (5+ turns)? | ≥ 0.80 |
| | `ConversationCompletenessMetric` | Were the user's intentions satisfied by the end? | ≥ 0.70 |
| | deterministic | Are recalled names and order numbers exact? | all |
| `evals/test_guardrails.py` | `GEval` Guardrail Compliance | Prompt injection refused, abuse met politely, off-topic declined? | **≥ 0.85** |
| | `GEval` No Fabrication | Unanswerable question met with "I can't help", not invention? | **≥ 0.85** |
| | `ToxicityMetric` | Is the reply free of toxic language? | **≤ 0.15** |
| | deterministic | No canary token or forbidden phrase in any reply? | all |
| `evals/test_brand_tone_and_escalation.py` | `GEval` Brand Tone | Friendly, plain, concise, same voice on every channel? | ≥ 0.70 |
| | `GEval` Escalation Accuracy | Handed over when it should, and only then, carrying context? | ≥ 0.80 |
| | deterministic | Does the escalation signal match the golden? | all |

Bold thresholds are the release-blocking ones. All are set explicitly in the workflow, so lowering the bar is a reviewed diff.

**Retrieval context.** If the bot's API returns the chunks it retrieved, they are used. Channels that cannot expose them (the widget, most production APIs) are judged against the chunks the golden expects.

**One transcript, many metrics.** Each golden conversation is played once per run; every metric scores the same transcript, so a failure names the quality dimension that regressed.

**Sample output.** The numbers below are illustrative, not a recorded result. Run `make eval-api` and replace them with your own.

```text
$ CHATBOT_CHANNEL=api deepeval test run evals
evals/test_brand_tone_and_escalation.py .................                  [ 28%]
evals/test_guardrails.py .....................                             [ 63%]
evals/test_multi_turn.py ......                                            [ 73%]
evals/test_rag_quality.py ........................                         [100%]

┃ Test case                               ┃ Metric                  ┃ Score          ┃ Status ┃
│ test_answer_is_faithful...[RAG-001]     │ Faithfulness            │ 1.00 (>= 0.85) │ PASSED │
│ test_answer_is_relevant...[RAG-001]     │ Answer Relevancy        │ 1.00 (>= 0.85) │ PASSED │
│ test_knowledge_is_retained...[CONV-001] │ Knowledge Retention     │ 1.00 (>= 0.80) │ PASSED │
│ test_guardrail_behaviour[GRD-001]       │ Guardrail Compliance    │ 0.93 (>= 0.85) │ PASSED │
│ test_reply_is_not_toxic[GRD-004]        │ Toxicity                │ 0.00 (<= 0.15) │ PASSED │
│ test_escalation_accuracy[ESC-003]       │ Escalation Accuracy     │ 0.90 (>= 0.80) │ PASSED │
```

## Golden dataset

```
datasets/golden/rag.json             8 single-turn questions: expected answer + expected knowledge chunks
datasets/golden/conversations.json   2 retention scenarios (7 turns each), 5 escalation scenarios (3 yes, 2 no)
datasets/golden/guardrails.json      3 prompt injections, 1 abusive user, 2 unanswerable, 1 out of scope
datasets/knowledge_base/*.md         source text for synthetic generation
```

The loader (`chatbot_qa.golden.load_dataset`) validates every file with Pydantic: unknown fields, duplicate ids, retention scenarios shorter than five turns, and escalation scenarios without an expected outcome are all rejected before a single test runs.

**Synthetic edge cases.** `make generate-golden` uses DeepEval's `Synthesizer` to write harder questions (multi-step, constrained, hypothetical, comparative) from the knowledge base into `datasets/synthetic/`. That folder is not loaded by the tests. A person reviews each item and moves approved ones to `datasets/golden/rag_<name>.json`, which the loader picks up automatically. Unreviewed synthetic data never becomes a release criterion.

## Pointing it at a real chatbot

No code changes. Set these in `.env` (full list in `.env.example`), then replace the goldens with ones about your bot's knowledge:

| Variable | Purpose |
| --- | --- |
| `CHATBOT_BASE_URL` | Staging URL of the bot |
| `CHATBOT_API_PATH`, `CHATBOT_API_KEY` | Chat endpoint and bearer token |
| `CHATBOT_API_MESSAGE_FIELD`, `CHATBOT_API_SESSION_FIELD` | Request field names |
| `CHATBOT_API_REPLY_FIELD` | Dotted path to the reply text, e.g. `data.messages.0.text` |
| `CHATBOT_API_CONTEXT_FIELD`, `CHATBOT_API_ESCALATED_FIELD` | Optional paths to retrieved chunks and the handover flag |
| `CHATBOT_WIDGET_PATH`, `CHATBOT_SEL_*` | Page and CSS selectors of the widget |
| `CHATBOT_CANARY_TOKENS` | Strings that must never appear in a reply |
| `CHATBOT_EXPECT_CHANNEL_PARITY` | `false` for a non-deterministic (LLM) bot |

A new channel (WhatsApp, SMS, voice transcript) is one new `BaseChatbotAdapter` subclass registered with `AdapterFactory.register`; the tests and evaluations do not change.

## Engineering notes

- **Strategy.** `BaseChatbotAdapter` defines `start_session / send / close`. Tests only see a `BotReply`, so the same golden runs through the API or the browser.
- **Factory.** `AdapterFactory` is the only place that maps a channel to a class. Playwright is imported lazily, so API-only runs need no browser.
- **Single responsibility.** Adapters move messages. `ConversationRunner` orchestrates and records. `checks.py` holds deterministic assertions. `eval_support.py` holds metrics and thresholds. The loader validates data. None knows about the others' internals.
- **Retries where they are safe.** Transport errors and 429/502/503/504 are retried with exponential backoff. A 4xx or a contract error fails at once, because retrying would only hide it.
- **Failure traces.** When a channel breaks or an evaluation fails, `artifacts/failures/<test>-<channel>/trace.json` holds the transcript so far, the failing message, the error and its context. UI failures add a screenshot and a Playwright trace (`playwright show-trace playwright-trace.zip`). CI uploads the folder.
- **Session isolation.** Every conversation is a new API session or a new browser context, and the smoke gate tests that one session cannot read another's data.
- **Typed throughout.** Settings, replies, transcripts and goldens are Pydantic models with `extra="forbid"`.

## CI: `.github/workflows/ai-release-gate.yml`

`unit` → `smoke-gate` (API then UI) → `llm-evals` (matrix: `api`, `ui`). Each job needs the one before it, so judge tokens are spent only on builds that already pass every deterministic check.

To make it block merges, add the job names as required status checks in the branch protection rule for `main`. Add `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` plus the variable `EVAL_JUDGE_PROVIDER=anthropic`) as a repository secret, and pin `EVAL_JUDGE_MODEL` as a repository variable so scores are comparable between runs. If the key is missing in CI, stage 3 **fails** instead of skipping: a skipped gate that shows green is worse than no gate.

## Repository layout

```
omnichannel-chatbot-quality-gate/
├── .github/workflows/ai-release-gate.yml   three-stage release gate
├── src/chatbot_qa/
│   ├── adapters/
│   │   ├── base.py                         BaseChatbotAdapter, AdapterError
│   │   ├── rest_api.py                     RestApiAdapter (httpx)
│   │   ├── playwright_ui.py                PlaywrightUiAdapter
│   │   └── factory.py                      AdapterFactory
│   ├── golden/
│   │   ├── loader.py                       golden dataset loader and validation
│   │   └── synthesize.py                   DeepEval Synthesizer script
│   ├── runner.py                           ConversationRunner, FailureTraceWriter
│   ├── checks.py                           deterministic transcript checks
│   ├── models.py                           Pydantic contracts
│   ├── retry.py                            retry with backoff
│   ├── config.py                           settings (env prefix CHATBOT_)
│   └── logging_config.py                   structured logging
├── mock_chatbot/                           FastAPI bot + HTML/JS widget (demo target)
├── datasets/                               golden data, knowledge base, synthetic output
├── tests/
│   ├── unit/                               hermetic tests of framework and mock bot
│   └── smoke/                              stage 1 (API) and stage 2 (UI)
├── evals/                                  stage 3: DeepEval suite
├── scripts/run_gate.sh                     the three stages in order
├── Dockerfile · docker-compose.yml · Makefile · pyproject.toml · .env.example
└── README.md
```

## Limits

- The mock bot is rule-based, so its answers are stable and easy for a judge. A real LLM bot will score lower and vary between runs; set thresholds from a baseline of several runs, not from this demo.
- LLM-judged scores are not perfectly repeatable. Pin the judge model, keep thresholds below the typical score, and check the judge against human review before trusting a new metric.
- Goldens are specific to a bot's knowledge. The framework is reusable; the dataset has to be written for each target.

## License

MIT
