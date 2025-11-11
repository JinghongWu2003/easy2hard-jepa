PYTHON ?= python
CONFIG ?= configs/stl10_baseline.yaml
CHECKPOINT ?= outputs/latest/checkpoints/last.pt
RUN_NAME ?=

setup:
$(PYTHON) -m pip install -r requirements.txt

pretrain:
$(PYTHON) scripts/pretrain.py --config $(CONFIG) $(if $(RUN_NAME),--run-name $(RUN_NAME),)

probe:
$(PYTHON) scripts/linear_probe.py --checkpoint $(CHECKPOINT)

knn:
$(PYTHON) scripts/knn_eval.py --checkpoint $(CHECKPOINT)

tsne:
$(PYTHON) scripts/visualize_tsne.py --checkpoint $(CHECKPOINT)

umap:
$(PYTHON) scripts/visualize_umap.py --checkpoint $(CHECKPOINT)

smoke:
$(PYTHON) scripts/smoke_test.py --config configs/stl10_small_smoketest.yaml

.PHONY: setup pretrain probe knn tsne umap smoke
