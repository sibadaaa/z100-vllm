#!/usr/bin/env python3
"""c=1 arm probe: ``s20_single_probe.py`` with the service API key attached.

The engine-step metric (``srv_step_ms`` from ``iteration_tokens_total``) is the
only c=1 quantity the S23 plan lets us claim, and it already exists in
``s20_single_probe.py``.  This entry point exists purely so the identical code
can talk to the keyed production service: it patches ``urlopen`` for the
``/v1/completions`` request and then runs the S20 CLI unchanged.

  .venv/bin/python tools/dcu/s23_probe.py --repeat 4 --tag control \
      --out .s23/w1/O6/w1a/control

Every argument is forwarded verbatim; see ``s20_single_probe.py --help``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import s23_common  # noqa: E402


def main():
    key = s23_common.install_auth()
    if not key:
        print("[s23_probe] WARN no API key found; /v1/* will 401", file=sys.stderr)

    import s20_single_probe

    return s20_single_probe.main()


if __name__ == "__main__":
    raise SystemExit(main())
