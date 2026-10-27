#!/usr/bin/env python3
"""c=4/c=16 arm sweep: ``s20_conc_sweep.py`` with the service API key attached.

Same rationale as ``s23_probe.py``: the sweep logic and metric definitions are
the ones recorded in the S20 report, so the O6 arms reuse it verbatim rather
than re-deriving step period / forward t/s.  Arguments are forwarded unchanged.

  .venv/bin/python tools/dcu/s23_sweep.py --concurrency 4,16 --tag o1
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import s23_common  # noqa: E402


def main():
    s23_common.install_auth()

    import s20_conc_sweep

    return s20_conc_sweep.main()


if __name__ == "__main__":
    raise SystemExit(main())
