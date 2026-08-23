# Attention-Lite canonical payload v2

This directory carries the repository-bundled canonical source artifact used by
`experiments.attention_lite_v1.source_resolver`.

## XSET source

`xset_01.b64` through `xset_04.b64`, concatenated in numeric order, are a
base64-encoded gzip stream containing the validated self-contained XSET E40
Attention-Lite Python source.

The resolver decodes the artifact, validates the locked architecture fingerprint
(T16, D128, Attention D64/H4/Dh16, 2,381,028 parameters, 705 leaves), validates
the official XSET split/schedule constants, and compiles the Python text before
it can be launched.

## XSUB source

The XSUB executable source is materialized deterministically from the decoded
XSET source. Only protocol-specific text, split sizes, and E40 schedule constants
outside `BUNDLE_B64` are changed to their validated XSUB values:

- train: 63,026
- validation: 50,919
- microsteps/epoch: 1,970
- total microsteps: 78,800
- optimizer schedule steps: 19,700

The embedded v4.1 `BUNDLE_B64` is preserved byte-for-byte. The resolver asserts
that the extracted bundle is identical between generated XSUB and XSET sources,
then validates and compiles both files.

Run the static integration check with:

```bash
python -m experiments.attention_lite_v1.canonical_selftest
```
