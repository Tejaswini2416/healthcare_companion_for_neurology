"""
Biophysical In-Silico Simulation Module
3D Reaction-Diffusion Fisher-Kolmogorov PDE Solver:
  dc/dt = ∇ · (D * ∇c) + ρ * c * (1 - c) - k_kill * c
Calibrated by IDH mutation status (proliferation ρ) and MGMT promoter methylation (cell kill k_kill).
Counterfactual forecasting across therapeutic regimens ('SOC_TMZ', 'DOSE_DENSE', 'HOLD_TMZ', 'LOMUSTINE')
over 30, 60, 90, and 180-day horizons with tumor volume and Karnofsky Performance Scale (KPS) trajectories.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class FisherKolmogorovSolver:
    """
    Biophysical reaction-diffusion solver for glioma invasion and growth.
    """
    def __init__(self):
        # Baseline physiological parameters (literature standard for human glioblastoma)
        # Spatial units: cm, Time units: days
        self.D_white_matter = 0.0013  # cm^2/day (~0.13 mm^2/day)
        self.detection_threshold = 0.16 # MRI T2/FLAIR detection threshold cell density
        self.core_threshold = 0.80      # Dense tumor core detection threshold

    def calibrate_parameters(
        self,
        idh_status: str,
        mgmt_methylated: bool,
        intervention: str = "SOC_TMZ"
    ) -> Tuple[float, float, float]:
        """
        Calibrates biophysical parameters:
          - D: Diffusivity / cell motility (cm^2/day)
          - rho (ρ): Proliferation rate (1/day)
          - k_kill: Therapeutic cell death rate (1/day)
        """
        # 1. Proliferation (ρ) calibrated by IDH Status
        if idh_status.lower() in ["mutant", "idh-mutant"]:
            # IDH mutation impairs tumor metabolism; slower doubling time
            rho = 0.016 # ~43 day doubling time
            D = 0.0009
        else:
            # IDH wild-type; aggressive proliferation
            rho = 0.048 # ~14 day doubling time
            D = 0.0016

        # 2. Cell Kill Rate (k_kill) calibrated by MGMT Methylation & Regimen
        # MGMT promoter methylation silences DNA repair, making cells vulnerable to alkylating agents
        base_sensitivity = 1.0 if mgmt_methylated else 0.28

        if intervention == "SOC_TMZ":
            # Standard 5/28 Temozolomide adjuvant cycle
            k_kill = 0.024 * base_sensitivity
        elif intervention == "DOSE_DENSE":
            # 7/14 or 21/28 dose-dense TMZ: higher drug concentration
            k_kill = 0.034 * base_sensitivity
        elif intervention == "HOLD_TMZ":
            # Therapy pause / holiday
            k_kill = 0.000
        elif intervention == "LOMUSTINE":
            # Second-line nitrosourea (CCNU)
            k_kill = 0.021 * (base_sensitivity * 0.85 + 0.15)
        else:
            k_kill = 0.015 * base_sensitivity

        return D, rho, k_kill

    def solve_radial_pde(
        self,
        initial_vol_cm3: float,
        D: float,
        rho: float,
        k_kill: float,
        total_days: int = 180,
        dt: float = 0.25,
        dr: float = 0.05, # cm
        max_r: float = 6.0 # cm
    ) -> Dict[int, float]:
        """
        Solves 3D spherically symmetric reaction-diffusion Fisher-Kolmogorov PDE:
          ∂c/∂t = D * (∂²c/∂r² + (2/r) * ∂c/∂r) + ρ * c * (1 - c) - k_kill * c
        Returns dictionary mapping day -> tumor volume in cm³.
        """
        # Initial radius from volume: V = (4/3) * pi * R^3 => R = (3V / 4pi)^(1/3)
        initial_radius = (3.0 * initial_vol_cm3 / (4.0 * math.pi)) ** (1.0 / 3.0)

        r_grid = np.arange(dr, max_r + dr, dr)
        N = len(r_grid)

        # Initial cell density profile c(r, 0): Gaussian-like tumor profile
        sigma = max(0.2, initial_radius / 1.6)
        c = np.exp(-0.5 * (r_grid / sigma) ** 2)
        c = np.clip(c, 0.0, 1.0)

        # Time-stepping
        num_steps = int(total_days / dt)
        checkpoints = {30, 60, 90, 180}
        trajectory = {0: initial_vol_cm3}

        current_day = 0.0
        for step in range(1, num_steps + 1):
            current_day += dt

            # Spatial derivatives with Neumann boundary conditions
            d2c_dr2 = np.zeros(N)
            dc_dr = np.zeros(N)

            # Center condition r -> 0: ∂c/∂r = 0, so (2/r)*∂c/∂r -> 2*∂²c/∂r²
            dc_dr[0] = (c[1] - c[0]) / dr
            d2c_dr2[0] = (c[1] - 2 * c[0] + c[0]) / (dr ** 2)

            # Interior nodes
            dc_dr[1:-1] = (c[2:] - c[:-2]) / (2 * dr)
            d2c_dr2[1:-1] = (c[2:] - 2 * c[1:-1] + c[:-2]) / (dr ** 2)

            # Outer boundary
            dc_dr[-1] = (c[-1] - c[-2]) / dr
            d2c_dr2[-1] = (c[-1] - 2 * c[-1] + c[-2]) / (dr ** 2)

            # Diffusion term: D * (d²c/dr² + (2/r)*dc/dr)
            diff_term = D * (d2c_dr2 + (2.0 / r_grid) * dc_dr)

            # Reaction term: ρ * c * (1 - c) - k_kill * c
            reaction_term = rho * c * (1.0 - c) - k_kill * c

            # Update density
            c = c + dt * (diff_term + reaction_term)
            c = np.clip(c, 0.0, 1.0)

            # Sample on milestone days
            day_int = int(round(current_day))
            if abs(current_day - day_int) < (dt / 2.0) and day_int in checkpoints and day_int not in trajectory:
                # Integrate volume where c(r) >= detection_threshold
                detected_mask = c >= self.detection_threshold
                if np.any(detected_mask):
                    detected_radius = r_grid[detected_mask][-1]
                    vol = (4.0 / 3.0) * math.pi * (detected_radius ** 3)
                else:
                    vol = max(0.5, float(initial_vol_cm3 * math.exp((rho - k_kill) * day_int)))
                trajectory[day_int] = round(max(0.2, vol), 2)

        return trajectory

    def compute_kps_trajectory(
        self,
        baseline_kps: int,
        initial_vol: float,
        trajectory_vols: List[float],
        intervention: str
    ) -> List[int]:
        """
        Computes patient functional performance (Karnofsky Performance Scale, 10-100)
        based on tumor mass burden and regimen-specific systemic toxicity.
        """
        kps_list = []
        for i, vol in enumerate(trajectory_vols):
            # Tumor volume impact
            vol_ratio = vol / max(0.1, initial_vol)
            mass_penalty = 0
            if vol_ratio > 1.5:
                mass_penalty = int((vol_ratio - 1.5) * 20)
            elif vol_ratio > 1.15:
                mass_penalty = 10

            # Regimen toxicity impact
            tox_penalty = 0
            if intervention == "DOSE_DENSE":
                tox_penalty = 5 if i > 0 else 0
                if i >= 3:
                    tox_penalty = 10 # cumulative fatigue
            elif intervention == "LOMUSTINE":
                tox_penalty = 5 if i >= 2 else 0

            # Treatment pause benefits short-term fatigue but suffers mass burden
            if intervention == "HOLD_TMZ" and i == 1:
                mass_penalty = max(0, mass_penalty - 5)

            score = baseline_kps - mass_penalty - tox_penalty
            # Quantize KPS to multiples of 5 between 10 and 100
            score = int(round(score / 5.0) * 5)
            score = max(20, min(100, score))
            kps_list.append(score)

        return kps_list

    def simulate_counterfactual(
        self,
        initial_vol_cm3: float,
        baseline_kps: int,
        idh_status: str,
        mgmt_methylated: bool,
        intervention: str = "SOC_TMZ",
        forecast_horizon_days: int = 180
    ) -> Dict[str, Any]:
        """
        Executes full in-silico simulation and returns time-series trajectory,
        KPS projection, and clinical guidance.
        """
        D, rho, k_kill = self.calibrate_parameters(idh_status, mgmt_methylated, intervention)

        # Solve reaction-diffusion PDE
        vol_dict = self.solve_radial_pde(
            initial_vol_cm3=initial_vol_cm3,
            D=D,
            rho=rho,
            k_kill=k_kill,
            total_days=forecast_horizon_days
        )

        milestones = [0, 30, 60, 90, 180]
        milestones = [d for d in milestones if d <= forecast_horizon_days]

        vols = []
        for d in milestones:
            if d in vol_dict:
                vols.append(vol_dict[d])
            else:
                # Analytic fallback estimate
                est = initial_vol_cm3 * math.exp((rho - k_kill) * d)
                vols.append(round(max(0.5, est), 2))

        # Compute KPS trajectory
        kps_scores = self.compute_kps_trajectory(baseline_kps, initial_vol_cm3, vols, intervention)

        # Confidence intervals (uncertainty widens over longer temporal horizons)
        confidence_intervals = [
            round(max(0.60, 0.95 - 0.0012 * d), 2) for d in milestones
        ]

        # Clinical interpretation summary
        delta_pct = round(((vols[-1] - initial_vol_cm3) / initial_vol_cm3) * 100.0, 1)
        if delta_pct < -10:
            regimen_summary = (
                f"Projected favorable tumor reduction of {abs(delta_pct)}% over {forecast_horizon_days} days. "
                "Maintains KPS functional stability with high probability of durable control."
            )
        elif delta_pct <= 10:
            regimen_summary = (
                f"Projected stable disease (Δ {delta_pct:+}%) over {forecast_horizon_days} days. "
                "Tumor proliferation and therapeutic cell kill are in equilibrium."
            )
        else:
            regimen_summary = (
                f"Projected progression of {delta_pct:+}% expansion over {forecast_horizon_days} days. "
                "Functional KPS projected to decline; alternative regimen or clinical trial evaluation advised."
            )

        return {
            "intervention": intervention,
            "forecast_horizon_days": forecast_horizon_days,
            "biophysical_parameters": {
                "D_motility_cm2_day": D,
                "diffusivity_cm2_day": D,
                "rho_proliferation_day": rho,
                "k_kill_day": k_kill,
                "net_growth_rate": round(rho - k_kill, 5),
                "net_growth_rate_day": round(rho - k_kill, 5),
            },
            "trajectory": {
                "days": milestones,
                "tumor_vol_cm3": vols,
                "kps_score": kps_scores,
                "confidence_interval": confidence_intervals,
            },
            "summary": regimen_summary,
        }
