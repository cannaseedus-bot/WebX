Micronaut-model skill (local)

Overview
- Scaffolds new micronauts (.xjson + bots.py) and copies a canonical python bridge (micronaut_native.py) into new micronaut folders.
- bots_template.py includes a native bridge import so generated bots automatically use TodoCreator, DeterministicV6, and TraceLogger when available.

Files to check
- assets/bots_template.py
- assets/micronaut_native.py
- scripts/scaffold_micronaut.py

Testing
- From repo root run: python -m pytest tests/test_micronaut_model_assets.py -q
