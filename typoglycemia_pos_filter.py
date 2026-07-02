"""
Compatibility wrapper for the POS-filtered typoglycemia attack.

New code should import PosFilteredTypoglycemia from pos_filter, or Typoglycemia
from typoglycemia for the unfiltered baseline implementation.
"""

from pos_filter import PosFilteredTypoglycemia
from pos_filter import Typoglycemia
from pos_filter import main

__all__ = ["PosFilteredTypoglycemia", "Typoglycemia", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
