"""Glue code that needs NeMo, PyTorch and a GPU.

NOTE: nothing in this sub-package could be executed in the environment where
it was written (no NeMo, no torch). Run `python -m scriptleak.nemo_tools.check_env`
first on the training machine; it verifies every NeMo entry point used here.
"""
