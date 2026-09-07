"""Internal single-protocol worker for the shared-cache corrected runner.

Launch both workers through run_dual_t4_corrected or kaggle_cell.py. The parent
prepares the shared cache before either worker imports JAX or opens a dataset.
"""
from experiments.nestsar_sm_all_t16.streaming.worker import main

if __name__ == "__main__":
    main()
