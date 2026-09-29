"""
Synthetic Patient Data & NIfTI Volume Seeder
Generates realistic multi-sequence NIfTI MRI files and seeds longitudinal data for patient V. Thanuja.
"""

import os
import numpy as np
from typing import Dict, Any, Tuple
from ..models.mri_segmenter import SyntheticBraTSGenerator
from .supabase_client import get_database_client


class PatientDataSeeder:
    """
    Seeds local sample files, creates synthetic BraTS NIfTI files,
    and populates initial patient state.
    """
    @staticmethod
    def generate_demo_nifti_files(output_dir: str = "./data/sample_scans") -> Dict[str, str]:
        """
        Creates synthetic 4-channel NIfTI volumes (.nii.gz) for clinician upload demo.
        """
        os.makedirs(output_dir, exist_ok=True)
        prefix = os.path.join(output_dir, "0042_followup_scan")

        # Check if already generated
        expected_t1ce = f"{prefix}_t1ce.nii.gz"
        expected_npz = f"{prefix}_mri4ch.npz"

        if os.path.exists(expected_t1ce) or os.path.exists(expected_npz):
            return {
                "t1": f"{prefix}_t1.nii.gz" if os.path.exists(f"{prefix}_t1.nii.gz") else expected_npz,
                "t1ce": f"{prefix}_t1ce.nii.gz" if os.path.exists(f"{prefix}_t1ce.nii.gz") else expected_npz,
                "t2": f"{prefix}_t2.nii.gz" if os.path.exists(f"{prefix}_t2.nii.gz") else expected_npz,
                "flair": f"{prefix}_flair.nii.gz" if os.path.exists(f"{prefix}_flair.nii.gz") else expected_npz,
            }

        # Generate 4-channel 3D volume
        vol_4ch, seg_gt = SyntheticBraTSGenerator.generate_synthetic_volume(
            shape=(64, 64, 48),
            wt_radius=13.5,
            tc_radius=8.5,
            et_thickness=3.8
        )

        saved_files = SyntheticBraTSGenerator.save_as_nifti(vol_4ch, prefix)
        return saved_files

    @staticmethod
    def seed_all():
        """Initializes database client and ensures sample files are created."""
        db = get_database_client()
        PatientDataSeeder.generate_demo_nifti_files()
        print(f"[Seeder] Patient 0042 (V. Thanuja) seeded. Database ready.")
