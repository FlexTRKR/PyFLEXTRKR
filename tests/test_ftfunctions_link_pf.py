"""
Regression test for an off-by-one in link_pf_tb (pyflextrkr/ftfunctions.py).

link_pf_tb loops `for ipf in range(1, npf)`, where npf = np.nanmax(pf_number).
pf_number is always produced by sort_renumber (label 1 = largest surviving PF,
labels increasing = decreasing size), so `range(1, npf)` silently skips
ipf == npf: the smallest surviving PF, on every frame. That PF's cloud
fragments are never merged to a common cloud number, and its unlabeled
no-cloud area is never filled in - live on every config with linkpf: 1.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyflextrkr.ftfunctions import link_pf_tb


def test_smallest_pf_is_linked():
    """
    Construct a domain with two PFs. PF 2 (the smaller one, i.e. ipf == npf)
    overlaps two separate cloud numbers that must be merged to the larger one.
    Before the fix, PF 2 is never visited, so the two clouds stay separate.
    """
    ny, nx = 10, 20
    tb = np.full((ny, nx), 280.0)

    # PF 1 (larger): columns 0-9, single cloud already merged (nothing to test here).
    pf_number = np.zeros((ny, nx), dtype=int)
    pf_number[2:8, 0:10] = 1

    # PF 2 (smaller, npf == 2, the one the off-by-one skips): columns 12-19,
    # covered by two distinct cloud numbers that should be merged to the larger.
    pf_number[3:7, 12:20] = 2

    cloudnumber = np.zeros((ny, nx), dtype=int)
    convcold_cloudnumber = np.zeros((ny, nx), dtype=int)
    # Cloud 1 inside PF1
    cloudnumber[3:6, 2:6] = 1
    convcold_cloudnumber[3:6, 2:6] = 1
    # Cloud 2 (larger, 16 px) and Cloud 3 (smaller, 4 px) both inside PF2
    cloudnumber[3:7, 12:16] = 2  # 4x4 = 16 px
    convcold_cloudnumber[3:7, 12:16] = 2
    cloudnumber[4:6, 17:19] = 3  # 2x2 = 4 px
    convcold_cloudnumber[4:6, 17:19] = 3

    pf_convcold_cloudnumber, pf_cloudnumber = link_pf_tb(
        convcold_cloudnumber, cloudnumber, pf_number, tb, tb_thresh=241.0,
    )

    # Within PF2, both cloud 2 and cloud 3 must be renumbered to the larger
    # (cloud 2, 16 px > cloud 3, 4 px).
    labels_in_pf2 = np.unique(pf_convcold_cloudnumber[pf_number == 2])
    labels_in_pf2 = labels_in_pf2[labels_in_pf2 > 0]
    assert len(labels_in_pf2) == 1, (
        f"PF 2 (npf, the smallest PF) still has multiple distinct cloud "
        f"numbers {labels_in_pf2} - link_pf_tb's ipf loop skipped it."
    )
    assert labels_in_pf2[0] == 2, (
        f"Expected clouds within PF 2 merged to the larger cloud number (2), "
        f"got {labels_in_pf2[0]}."
    )

    # Same check for pf_cloudnumber.
    labels_in_pf2_cn = np.unique(pf_cloudnumber[pf_number == 2])
    labels_in_pf2_cn = labels_in_pf2_cn[labels_in_pf2_cn > 0]
    assert len(labels_in_pf2_cn) == 1
    assert labels_in_pf2_cn[0] == 2


def test_larger_pf_still_linked_when_smaller_pf_also_present():
    """
    Sanity check: with npf >= 2, ipf == 1 (the largest PF) is inside
    range(1, npf) either way, so it must merge correctly regardless of the
    off-by-one. This isolates the bug to specifically the ipf == npf case,
    rather than PF linking being broken in general.
    """
    ny, nx = 10, 20
    tb = np.full((ny, nx), 280.0)
    pf_number = np.zeros((ny, nx), dtype=int)
    pf_number[2:8, 0:10] = 1  # larger PF (ipf=1, always in range regardless of bug)
    pf_number[3:7, 12:20] = 2  # smaller PF (ipf=npf=2, only in range with the fix)

    cloudnumber = np.zeros((ny, nx), dtype=int)
    convcold_cloudnumber = np.zeros((ny, nx), dtype=int)
    cloudnumber[2:5, 2:5] = 1  # 9 px, inside PF 1
    convcold_cloudnumber[2:5, 2:5] = 1
    cloudnumber[5:8, 5:8] = 2  # 9 px, inside PF 1 - tie, either wins, must merge
    convcold_cloudnumber[5:8, 5:8] = 2
    cloudnumber[4:6, 14:16] = 3  # inside PF 2, single cloud, nothing to merge
    convcold_cloudnumber[4:6, 14:16] = 3

    pf_convcold_cloudnumber, pf_cloudnumber = link_pf_tb(
        convcold_cloudnumber, cloudnumber, pf_number, tb, tb_thresh=241.0,
    )
    labels = np.unique(pf_convcold_cloudnumber[pf_number == 1])
    labels = labels[labels > 0]
    assert len(labels) == 1, f"PF 1 clouds not merged: {labels}"


def test_single_pf_domain_is_still_linked():
    """
    With exactly one PF, npf == 1, so range(1, npf) == range(1, 1) == empty:
    the bug means link_pf_tb does nothing at all in the single-PF case, the
    most common one in practice. This must not regress after the fix.
    """
    ny, nx = 10, 10
    tb = np.full((ny, nx), 280.0)
    pf_number = np.zeros((ny, nx), dtype=int)
    pf_number[2:8, 2:8] = 1

    cloudnumber = np.zeros((ny, nx), dtype=int)
    convcold_cloudnumber = np.zeros((ny, nx), dtype=int)
    cloudnumber[2:5, 2:5] = 1  # 9 px
    convcold_cloudnumber[2:5, 2:5] = 1
    cloudnumber[5:8, 5:8] = 2  # 9 px, tie - either wins, just must merge
    convcold_cloudnumber[5:8, 5:8] = 2

    pf_convcold_cloudnumber, pf_cloudnumber = link_pf_tb(
        convcold_cloudnumber, cloudnumber, pf_number, tb, tb_thresh=241.0,
    )
    labels = np.unique(pf_convcold_cloudnumber[pf_number == 1])
    labels = labels[labels > 0]
    assert len(labels) == 1, f"PF 1 clouds not merged: {labels}"


if __name__ == "__main__":
    test_smallest_pf_is_linked()
    test_larger_pf_still_linked_when_smaller_pf_also_present()
    test_single_pf_domain_is_still_linked()
    print("All link_pf_tb tests passed.")
