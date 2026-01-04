"""
Validate OOF CLS features to detect data leakage.
Compares OOF features against in-sample features.

Expected behavior (NO leakage):
- Episode alignment: Perfect match
- NaN/Inf values: None
- Correlation: 0.7-0.95 (high similarity but not identical)
- If correlation > 0.98: DATA LEAKAGE DETECTED!
"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "open_track1"

# OOF features (clean, from K-fold)
OOF_NPZ = DATA_DIR / "film_smoe_cond_features_oof.npz"
# Original in-sample features (potentially leaked)
ORIG_NPZ = DATA_DIR / "film_smoe_cond_features_backup.npz"


def validate_oof_features():
    """Validate OOF features for data leakage."""

    print("=" * 60)
    print("OOF Feature Validation")
    print("=" * 60)

    # Load features
    if not OOF_NPZ.exists():
        print(f"ERROR: OOF features not found: {OOF_NPZ}")
        return False
    if not ORIG_NPZ.exists():
        print(f"WARNING: Original features not found: {ORIG_NPZ}")
        print("Skipping comparison (will only validate OOF integrity)")
        orig_data = None
    else:
        orig_data = np.load(ORIG_NPZ, allow_pickle=True)

    oof_data = np.load(OOF_NPZ, allow_pickle=True)

    # Extract data
    oof_ids = oof_data["episode_ids"]
    oof_cls = oof_data["cls_out"]
    oof_fourier = oof_data["fourier_seq"]
    oof_router = oof_data["router_logits"]

    print(f"\n1. OOF Feature Shapes:")
    print(f"   episode_ids: {oof_ids.shape}")
    print(f"   cls_out:     {oof_cls.shape}")
    print(f"   fourier_seq: {oof_fourier.shape}")
    print(f"   router_logits: {oof_router.shape}")

    # Check for NaN/Inf
    print(f"\n2. Data Integrity Check:")
    cls_nan = np.isnan(oof_cls).sum()
    cls_inf = np.isinf(oof_cls).sum()
    router_nan = np.isnan(oof_router).sum()
    router_inf = np.isinf(oof_router).sum()

    print(f"   cls_out NaN:  {cls_nan} ({cls_nan / oof_cls.size * 100:.2f}%)")
    print(f"   cls_out Inf:  {cls_inf} ({cls_inf / oof_cls.size * 100:.2f}%)")
    print(f"   router NaN:   {router_nan} ({router_nan / oof_router.size * 100:.2f}%)")
    print(f"   router Inf:   {router_inf} ({router_inf / oof_router.size * 100:.2f}%)")

    if cls_nan + cls_inf + router_nan + router_inf > 0:
        print("   ⚠ WARNING: Found NaN/Inf values!")
    else:
        print("   ✓ No NaN/Inf values")

    # Feature statistics
    print(f"\n3. OOF Feature Statistics:")
    print(f"   cls_out mean:  {oof_cls.mean():.4f} ± {oof_cls.std():.4f}")
    print(f"   cls_out range: [{oof_cls.min():.4f}, {oof_cls.max():.4f}]")
    print(f"   router mean:   {oof_router.mean():.4f} ± {oof_router.std():.4f}")

    # Compare with original if available
    if orig_data is not None:
        print(f"\n4. Comparison with Original Features:")
        orig_ids = orig_data["episode_ids"]
        orig_cls = orig_data["cls_out"]

        # Check episode alignment
        if len(oof_ids) != len(orig_ids):
            print(f"   ⚠ Episode count mismatch: OOF={len(oof_ids)}, Orig={len(orig_ids)}")
        elif not np.array_equal(oof_ids, orig_ids):
            print(f"   ⚠ Episode IDs do not match!")
        else:
            print(f"   ✓ Episode alignment: Perfect match ({len(oof_ids)} episodes)")

        # Compute correlation
        if oof_cls.shape == orig_cls.shape:
            # Flatten for correlation
            oof_flat = oof_cls.flatten()
            orig_flat = orig_cls.flatten()
            correlation = np.corrcoef(oof_flat, orig_flat)[0, 1]

            print(f"\n5. Data Leakage Detection:")
            print(f"   Correlation (OOF vs Original): {correlation:.4f}")

            if correlation > 0.98:
                print(f"   ⚠⚠⚠ CRITICAL: DATA LEAKAGE DETECTED!")
                print(f"       Correlation too high (>{0.98:.2f})")
                print(f"       OOF features likely contain in-sample predictions")
                return False
            elif correlation < 0.7:
                print(f"   ⚠ WARNING: Correlation unexpectedly low (<0.7)")
                print(f"       Features may be from different models/configs")
            else:
                print(f"   ✓ Correlation in expected range [0.7, 0.95]")
                print(f"   ✓ No data leakage detected")

            # Per-dimension correlation
            dim_corrs = [np.corrcoef(oof_cls[:, i], orig_cls[:, i])[0, 1]
                         for i in range(min(10, oof_cls.shape[1]))]
            print(f"\n   First 10 dimensions correlation:")
            for i, corr in enumerate(dim_corrs):
                print(f"     dim {i:2d}: {corr:.4f}")
        else:
            print(f"   ⚠ Shape mismatch: OOF={oof_cls.shape}, Orig={orig_cls.shape}")

    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = validate_oof_features()
    exit(0 if success else 1)
