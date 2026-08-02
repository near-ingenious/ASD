"""
asd_multimodal — Robust ASD Diagnosis Under Missing Clinical Modalities
========================================================================
Cross-Modal Representation Reconstruction and Explainable Multimodal Learning

Authors:
    Jarin Alam Prity  (222-115-005)  jarinprity438@gmail.com
    Popy Rani Boidya  (007)          popyboidya@gmail.com

Supervisor:
    Md Mahfujul Hasan, Associate Professor & Head, Dept. of CSE
    Metropolitan University, Sylhet-3104, Bangladesh
    mahfujul@metrouni.edu.bd

Clinical Collaborator:
    Prof. Imdadul Magfur, Psychiatrist & Psychotherapist
    Sylhet MAG Osmani Medical College & Hospital
    imdad153153@gmail.com

Target Journal: IEEE Transactions on Medical Imaging (Q1)
Dataset:        ABIDE-I (n=989) + ABIDE-II (n=1,114)
"""
__version__  = "1.0.0"
__authors__  = [
    "Jarin Alam Prity",
    "Popy Rani Boidya",
]
__emails__   = [
    "jarinprity438@gmail.com",
    "popyboidya@gmail.com",
]
__institution__ = "Metropolitan University, Sylhet-3104, Bangladesh"
__supervisor__  = "Md Mahfujul Hasan"
__clinical__    = "Prof. Imdadul Magfur"
__license__     = "MIT"

from .utils.metrics import compute_metrics, bootstrap_ci, format_ci, sig_stars

__all__ = [
    "compute_metrics",
    "bootstrap_ci",
    "format_ci",
    "sig_stars",
    "__version__",
    "__authors__",
    "__institution__",
]
