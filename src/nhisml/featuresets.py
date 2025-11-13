# src/nhisml/featuresets.py
from dataclasses import dataclass
from typing import List

@dataclass
class FeatureSet:
    name: str
    binary_12: List[str]
    ordinal: List[str]
    categorical: List[str]

    @property
    def all_columns(self) -> List[str]:
        # include target + weight always
        base = ["PHSTAT_A", "WTFA_A"]
        return list({*self.binary_12, *self.ordinal, *self.categorical, *base})

def get_featureset(name: str = "srh_core") -> FeatureSet:
    """
    Curated SRH core feature set (train on 2023, evaluate on 2024).
    """
    if name != "srh_core":
        # You can extend with more named sets as needed
        raise ValueError(f"Unknown featureset: {name}")

    binary = [
        "EMPWRKFT1_A", "EMPHEALINS_A", "EMPSICKLV_A", "EMPLASTWK_A",
        "DISAB3_A", "DIFF_A", "COGMEMDFF_A", "VISIONDF_A", "HEARINGDF_A",
        "DEPMED_A", "ANXMED_A", "MHRX_A", "MHTHRPY_A", "MHTHDLY_A", "MHTHND_A",
        "HYPEV_A", "DIBEV_A", "CHDEV_A", "MIEV_A", "STREV_A", "ANGEV_A",
        "ASEV_A", "ASTILL_A", "ARTHEV_A", "COPDEV_A", "CANEV_A",
        "CHLEV_A", "CHL12M_A", "HYP12M_A", "HYPMED_A", "KIDWEAKEV_A", "LIVEREV_A",
        "HEPEV_A", "CROHNSEV_A", "ULCCOLEV_A", "PSOREV_A", "CFSNOW_A",
        "HICOV_A", "USUALPL_A", "MEDNG12M_A", "MEDDL12M_A", "RXDG12M_A",
        "EMDSUPER_A"
    ]

    ordinal = [
        "RATCAT_A", "POVRATTC_A", "EDUCP_A", "MAXEDUCP_A",
        "LONELY_A", "SUPPORT_A", "FDSCAT3_A", "FDSCAT4_A",
        "WORTHLESS_A", "HOPELESS_A", "SAD_A", "NERVOUS_A", "RESTLESS_A", "EFFORT_A",
        "DEPFREQ_A", "ANXFREQ_A", "DEPLEVEL_A",
        "EMPWKHRS3_A", "LASTDR_A", "WELLVIS_A"
    ]

    categorical = [
        "MARITAL_A", "MARSTAT_A", "URBRRL23", "REGION", "EMPNOWRK_A", "EMPWHENWRK_A"
    ]

    return FeatureSet("srh_core", binary_12=binary, ordinal=ordinal, categorical=categorical)
