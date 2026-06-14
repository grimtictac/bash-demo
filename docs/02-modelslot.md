# Issue: Model/Version Torn Read

## The Problem

The original service stored the active model and its version as two separate instance variables: `_current_model` and `_current_version`. A predict thread reading these two variables and a swap thread writing them could interleave.

```python
# original swap
self._current_model = new_model
self._current_version = new_version   # predict thread can read between these two lines
```

A predict thread that reads `_current_model` after the first assignment but `_current_version` before the second sees a new model paired with an old version string. In the worst case — where model interfaces change between versions — this could result in a request being passed to a model that does not expect it.

The original code added `_swap_lock` to guard against this: both the swap and every predict call had to acquire the lock before touching either variable. This protected against the race but serialised all predict calls behind every swap.

## The Fix

`ModelSlot` is a frozen dataclass that bundles model and version into a single reference:

```python
@dataclass(frozen=True)
class ModelSlot:
    model: ModelProtocol
    version: Optional[str]
```

A swap creates a new slot and assigns it in one step:

```python
self._slot = ModelSlot(new_model, self._model_version)
```

In CPython, reference assignment is atomic — a predict thread reading `self._slot` always gets either the old slot or the new slot, never a mixture. `_swap_lock` is eliminated entirely.

```python
# predict — no lock needed
slot = self._slot
```

The same approach applies to rollback. Previously, separate `_previous_model` and `_previous_version` variables had the same problem. Now a single `_previous_slot` reference replaces both.

## Proof

The refactor removes `_swap_lock` and reduces four state variables to two. The logic is simpler, not just different — there is no lock to forget to acquire, and no window between the two assignments.

The rollback path received dedicated test coverage to validate the single-clear semantics: after a rollback, `_previous_slot` is set to `None` rather than swapped, so a second rollback returns `False` instead of toggling back to a bad model.

The full test suite (62 tests) passes, covering load, rollback, concurrent access, and all HTTP endpoints.

## No Regressions

Removing `_swap_lock` has no effect on the `_load_lock` that serialises concurrent `load_model()` calls — that lock remains and is unrelated to this change.
