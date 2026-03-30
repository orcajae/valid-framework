"""Unit tests for CPCV."""
import numpy as np
from valid.cpcv import make_groups, cpcv_split, cpcv_paths


def test_make_groups():
    gids = make_groups(100, 5)
    assert len(gids) == 100
    assert len(set(gids)) == 5


def test_cpcv_paths():
    paths = cpcv_paths(6, 2)
    assert len(paths) == 15


def test_cpcv_split_purge():
    gids = make_groups(100, 5)
    train, test = cpcv_split(gids, (0, 1), purge_bars=5)
    assert len(test) == 40
    assert len(train) < 60  # some purged
    assert not np.any(np.isin(train, test))
