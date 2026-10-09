"""tools/check_commit_msg.py enforces COMMIT_POLICY.md."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_commit_msg", Path(__file__).resolve().parents[1] / "tools" / "check_commit_msg.py")
ccm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ccm)


@pytest.mark.parametrize("msg", [
    "fix(trainer): restore best weights for test eval",
    "docs: update tutorials index",
    "arch(lform): gate externals with D\n\n- Routed B u_t through D\n\nKeeps D -> 0 a pure memory.",
    "refactor(core)!: rename delta_term\n\n- Renamed it\n\nBREAKING CHANGE: use .correction",
    "feat(registry): build ensembles by name\n\n- Added build_ensemble\n\nCo-Authored-By: A <a@b.c>",
    "Merge pull request #10 from rock-feller/refactor/x",
    "# only a comment\nfix(ptf): clamp p\n# trailing comment",
])
def test_valid_messages(msg):
    errors, _ = ccm.check(msg)
    assert errors == []


@pytest.mark.parametrize("msg, fragment", [
    ("Added stuff", "header must look like"),
    ("feat (MSI-1000):add export", "header must look like"),
    ("Fix(ptf): clamp p", "header must look like"),
    ("feature(ptf): clamp p", "unknown type"),
    ("fix(ptf): clamp p.", "period"),
    ("fix(ptf): " + "x" * 70, "hard limit"),
    ("fix(ptf): clamp p\nbody right away", "blank line"),
    ("math(core): rescale the correction", "need a body"),
    ("", "empty"),
])
def test_invalid_messages(msg, fragment):
    errors, _ = ccm.check(msg)
    assert any(fragment in e for e in errors), errors


@pytest.mark.parametrize("msg, fragment", [
    ("fix(ptf): fixed the clamp", "imperative"),
    ("fix(ptf): Clamp p", "lowercase"),
    ("feat(registry): add a rather long summary line here", "aim for 50"),
    ("refactor(core)!: rename delta_term", "BREAKING CHANGE"),
])
def test_warnings(msg, fragment):
    errors, warnings = ccm.check(msg)
    assert not errors and any(fragment in w for w in warnings), warnings


def test_header_only_skips_body_rules():
    assert ccm.check("math(core): rescale the correction", header_only=True)[0] == []
    assert ccm.check("Rescale the correction", header_only=True)[0]
