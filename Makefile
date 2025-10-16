PYTHON ?= python3

.PHONY: bootstrap up down reset smoke deploy

bootstrap:
	$(PYTHON) scripts/run_local.py bootstrap

up:
	$(PYTHON) scripts/run_local.py up

down:
	$(PYTHON) scripts/run_local.py down --remove-orphans

reset:
	$(PYTHON) scripts/run_local.py reset --regenerate-env

smoke:
	$(PYTHON) scripts/run_local.py smoke

deploy: up
