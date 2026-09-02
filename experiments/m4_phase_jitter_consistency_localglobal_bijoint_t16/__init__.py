"""LocalGlobal V2 + bidirectional joint-memory spatial encoder.

Clean ablation from the verified LocalGlobal V2 champion. The only neural
architecture change is the 25-joint spatial memory: the original one-way
GatedSweep is replaced by the same bidirectional BiMemory primitive already
used by NestSAR temporally. No attention and no QKV are introduced.
"""
