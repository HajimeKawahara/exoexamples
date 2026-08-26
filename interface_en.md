# Ownership Principles

**Document status:** This document is an English translation of
[`interface_ja.md`](interface_ja.md). The Japanese version is authoritative if
the two versions differ in interpretation.

The ExoFamily projects use the following common rules.

## Principle A: A provider exposes its own physical quantities in a general form

ExoEOS exposes EOS states, density, fugacity, and activity.
ExoGibbs exposes species, mole fractions, and equilibrium states.
ExoJAX exposes pressure grids, atmospheric structures, opacity, and radiative
transfer.

A provider does not know about a specific consumer.

For example, ExoEOS does not import ExoGibbs or ExoJAX.

## Principle B: The consumer owns the port

ExoGibbs defines the fugacity callback it requires.
ExoJAX defines the density provider and composition input it requires.
ExoJAX defines the pressure-boundary contract it requires.

## Principle C: A pair-specific adapter exists only once, under the consumer's `interop`

### Naming convention

Use:

```text
<consumer>.interop.<provider>
```

For example:

```text
exogibbs.interop.exoeos
exojax.interop.exogibbs
exojax.interop.exoeos
```

If the packages can connect directly through a simple array contract, it is
better not to create a pair-specific adapter at all.

## Principle D: A core module does not import a sibling package

For example, importing ExoGibbs must not require ExoEOS:

```python
import exogibbs
```

Import an adapter only when it is actually used:

```python
from exogibbs.interop.exoeos import make_pure_lnphi_func
```

An `interop/__init__.py` file is empty or lazy, so a top-level import does not
load another package.
