"""Feature gate resolver for conditional feature inclusion.

This module implements the gating logic for features that are conditionally
included based on Phase 3's machine-readable gate contract. All gating decisions
are transparent and documented in feature_manifest.json.

IMPORTANT: Gate values are read exclusively from phase3_gating_decisions.yaml.
The engineering_findings.md document is human-readable only and must never be
parsed by this module.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict


class FeatureGateResolver:
    """Resolves which conditional features should be included based on Phase 3 gate contract."""

    def __init__(self, logger: logging.Logger):
        """Initialize gate resolver.

        Args:
            logger: Logger instance for recording gate decisions.
        """
        self.logger = logger
        self.gates = {
            "skewness": False,
            "rolling_variance": False,
            "envelope_features": False,
            "xy_energy_ratio": False,
        }
        self.gate_decisions = {}

    def resolve_gates(
        self, phase3_contract_path: Path, bearing_geometry_available: bool
    ) -> Dict[str, bool]:
        """Resolve all feature gates from the Phase 3 YAML gate contract.

        Args:
            phase3_contract_path: Path to phase3_gating_decisions.yaml
            bearing_geometry_available: Whether bearing geometry was verified in Phase 3.
                Overrides envelope_features gate to False when False, regardless of YAML.

        Returns:
            dict mapping gate names to bool inclusion decisions
        """
        self.logger.info("Resolving feature gates from Phase 3 gate contract")

        contract_gates = self._load_yaml_contract(phase3_contract_path)

        for gate_name in ["skewness", "rolling_variance", "envelope_features", "xy_energy_ratio"]:
            if gate_name in contract_gates:
                gate_entry = contract_gates[gate_name]
                include = bool(gate_entry.get("include", False))
                justification = gate_entry.get("justification", "From phase3_gating_decisions.yaml")
            else:
                include = False
                justification = f"Gate '{gate_name}' absent from contract; defaulting to EXCLUDE"

            self.gates[gate_name] = include
            self.gate_decisions[gate_name] = {
                "included": include,
                "reason": justification,
                "source": str(phase3_contract_path),
            }

        # Bearing Geometry Safety Rule: override envelope gate when geometry not available.
        # This rule is non-negotiable and takes precedence over the YAML contract value.
        if not bearing_geometry_available:
            self.gates["envelope_features"] = False
            self.gate_decisions["envelope_features"] = {
                "included": False,
                "reason": "Bearing Geometry Safety Rule",
                "source": "bearing_geometry.py verification failed in Phase 3",
            }

        for gate_name, decision in self.gate_decisions.items():
            inclusion_status = "INCLUDE" if decision["included"] else "EXCLUDE"
            self.logger.info(f"Gate [{gate_name}] = {inclusion_status}: {decision['reason']}")

        return self.gates

    def _load_yaml_contract(self, contract_path: Path) -> dict:
        """Load gate_decisions section from the Phase 3 YAML contract.

        Returns empty dict (all gates default to False) if file missing or unreadable.
        """
        if not contract_path.exists():
            self.logger.warning(
                f"Phase 3 gate contract not found at {contract_path}; "
                "all gates defaulting to EXCLUDE"
            )
            return {}

        try:
            with open(contract_path, "r", encoding="utf-8") as f:
                contract = yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(
                f"Failed to parse gate contract at {contract_path}: {e}; "
                "all gates defaulting to EXCLUDE"
            )
            return {}

        gate_decisions = contract.get("gate_decisions", {})
        if not gate_decisions:
            self.logger.warning(
                "Gate contract contains no gate_decisions section; all gates defaulting to EXCLUDE"
            )

        return gate_decisions

    def get_gate_decisions(self) -> Dict:
        """Return the current gate decision record.

        Returns:
            dict with gate resolutions and complete rationale
        """
        return self.gate_decisions
