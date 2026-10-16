PY ?= .venv/bin/python

artifacts:       ## download the trained ONNX model + embedding bank from the v1.0.0 release
	curl -fsSL -o artifacts/chipnet.onnx https://github.com/QuintonMcReynolds/geoint-triage/releases/download/v1.0.0/chipnet.onnx
	curl -fsSL -o artifacts/triage_bank.npz https://github.com/QuintonMcReynolds/geoint-triage/releases/download/v1.0.0/triage_bank.npz

data:            ## download + cache EuroSAT RGB
	$(PY) -c "from geoint.data import prepare; prepare()"
train:           ## train clean + degradation-augmented models
	$(PY) scripts/train.py --variant clean
	$(PY) scripts/train.py --variant robust
eval:            ## all offline metrics -> reports/results.json, figures, service artifacts
	$(PY) scripts/evaluate.py
export:          ## ONNX export + parity check
	$(PY) scripts/export_onnx.py robust
feed:            ## demo collection passes
	$(PY) scripts/make_feed.py
agent-eval:      ## 30-question agent benchmark for each local model
	for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct; do $(PY) scripts/eval_agent.py --model $$m; done
test:
	$(PY) -m pytest -q
serve:
	$(PY) -m uvicorn serve.app:app --port 8080
all: data train eval export feed test

.PHONY: artifacts data train eval export feed agent-eval test serve all
