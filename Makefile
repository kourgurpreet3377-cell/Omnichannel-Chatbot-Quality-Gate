.PHONY: install serve test lint smoke-api smoke-ui eval-api eval-ui gate demo demo-fault generate-golden clean

install:          ## install the framework, the mock bot and a Chromium browser
	pip install -e ".[dev]"
	playwright install chromium

serve:            ## run the mock chatbot on http://localhost:8080 (FAULT=hallucinate|forget|leak)
	MOCK_BOT_FAULT=$(or $(FAULT),none) uvicorn mock_chatbot.app:app --port 8080

test:             ## hermetic unit tests: no server, no browser, no keys
	pytest

lint:
	ruff check .

smoke-api:        ## stage 1: deterministic API gate (needs a running chatbot)
	pytest tests/smoke/test_api_smoke.py

smoke-ui:         ## stage 2: deterministic widget gate (needs a running chatbot)
	pytest tests/smoke/test_ui_smoke.py

eval-api:         ## stage 3 through the REST API adapter (needs a judge API key)
	CHATBOT_CHANNEL=api deepeval test run evals

eval-ui:          ## stage 3 through the Playwright widget adapter (needs a judge API key)
	CHATBOT_CHANNEL=ui deepeval test run evals

gate:             ## all three stages in order, against the chatbot at CHATBOT_BASE_URL
	bash scripts/run_gate.sh

demo:             ## one command: build, start the mock chatbot, run the whole gate in Docker
	docker compose up --build --abort-on-container-exit --exit-code-from gate

demo-fault:       ## same, with a regression injected: make demo-fault FAULT=leak
	MOCK_BOT_FAULT=$(or $(FAULT),hallucinate) docker compose up --build --abort-on-container-exit --exit-code-from gate

generate-golden:  ## synthesize edge-case goldens from datasets/knowledge_base (needs a judge API key)
	python -m chatbot_qa.golden.synthesize --per-context 2

clean:
	docker compose down --remove-orphans 2>/dev/null || true
	rm -rf artifacts .pytest_cache .ruff_cache .deepeval
