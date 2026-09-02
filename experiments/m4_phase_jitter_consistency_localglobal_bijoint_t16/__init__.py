"""LocalGlobal V2 + bidirectional joint-memory spatial encoder.

Clean architecture ablation:
- exact LocalGlobal V2 T16 preprocessing/training recipe;
- exact J/B/JM/BM streams, temporal BiMemory, router, descriptors, uniform fusion;
- ONLY change: the 25-joint spatial sweep becomes bidirectional using the same
  GatedSweep/BiMemory primitive already used elsewhere in NestSAR.
- no attention / no QKV.
"""
