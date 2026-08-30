"""Recovered historical result-generation code.

Nothing in this package is a reimplementation. Every function here was copied from the session
scripts that actually produced the published numbers, with computational semantics preserved
byte-for-byte in the parts that affect a result. Provenance for each is recorded in
``publication/repro/``.

Do not "clean up" the arithmetic in this package. Two of the recovered bootstrap call sites use a
different upper percentile index from the others, and that difference is historical fact rather than
a bug to be normalised away.
"""
