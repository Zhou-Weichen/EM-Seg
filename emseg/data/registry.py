"""Source registry - the single source of truth for every dataset in the corpus.

Each Source says WHAT the data is (content, organism, modality, pixel size, annotated classes, label quality, licence,
citation), WHERE it comes from (every URL with its destination under raw_downloads/ and, when the publisher gives one,
the expected sha256), HOW it is fetched (method) and converted (converter jobs), and WHICH project-specific rules apply
(benchmark-leakage exclusions, known duplicates, sub-sampling). The pipeline stages only read this file; nothing about
a dataset is hard-coded elsewhere except the format-specific reading code of its converter.

Download methods
  http      plain HTTP(S) files, fetched by tools/pdl.py (parallel range requests, resumable, size check, sha256)
  listing   HTTP files whose list is read from the server at download time (EMPIAR directory index, BioStudies file list)
  gdown     Google-Drive folders (gdown; quota-limited, retried)
  stream    no raw download: anonymous cloud-volume / zarr cut-outs written directly by the converter (connectomes, ASEM)
  local     in-house data on the read-only mount (not public; excluded from the default manifests)
Host policy: zenodo.org rate-limits parallel connections (403 "unusual traffic") -> 1 connection, one file at a
time; ftp.ebi.ac.uk refuses many connections -> 2; everything else 6 connections.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ZEN = "https://zenodo.org"
EBI = "https://ftp.ebi.ac.uk/empiar/world_availability"
GCS_FIB25 = "https://storage.googleapis.com/ffn-flyem-fib25"
EPFL = "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus"


@dataclass
class Source:
    id: str
    name: str
    content: str
    organism: str
    modality: str
    resolution_nm: str
    classes: list
    quality: str                      # gold = expert / proofread; silver = automatic (partly proofread)
    license: str
    citation: str
    homepage: str
    method: str                       # http | listing | gdown | stream | local
    files: list = field(default_factory=list)       # [(dest under raw_downloads/em or natural, url, sha256 or "", extra pdl flags)]
    converters: list = field(default_factory=list)  # ["module:job", ...] (emdata.<module>.JOBS[job] or module main)
    outputs: list = field(default_factory=list)     # dataset names written to em/, em2d/, natural/
    rules: str = ""                   # leakage / dedup / sub-sampling rules applied for this project
    default: bool = True              # part of the default corpus
    redistribution: str = "check"     # open | nc | check | restricted  (see REDISTRIBUTION below)


def F(dest, url, sha="", flags=""):
    return (dest, url, sha, flags)


SOURCES = [
    # ------------------------------------------------------------------ dense cell / neurite instances (expert)
    Source("snemi3d", "SNEMI3D (ISBI 2013), Kasthuri et al. 2015 AC4/AC3", "mouse somatosensory cortex neuropil, every neurite labelled",
           "mouse", "ATUM-SEM", "6x6x29", ["cell"], "gold", "challenge terms (research use)", "Kasthuri et al., Cell 2015",
           "https://snemi3d.grand-challenge.org/", "http",
           [F("em/snemi/snemi_train.h5", "https://oc.embl.de/index.php/s/43iMotlXPyAB39z/download", "5b130a24d9eb23d972fede0f1a403bc05f6808b361cfa22eff23b930b12f0615"),
            F("em/snemi/snemi_test.h5", "https://oc.embl.de/index.php/s/aRhphk35H23De2s/download", "3df3920a0ddec6897105845f842b2665d37a47c2d1b96d4f4565682e315a59fa")],
           ["local_mount:snemi"], ["snemi3d"],
           "only AC4 z0-79 is used: AC4 z80-99 is this project's validation set and AC3 its test set; "
           "the AC3 file is downloaded only for the leakage scan"),
    Source("cremi", "CREMI 2016 samples A/B/C", "adult drosophila brain neuropil (FAFB region), neuron ids", "drosophila", "ssTEM", "4x4x40",
           ["cell"], "gold", "CC-BY 4.0", "Funke et al., CREMI challenge 2016", "https://cremi.org/data/", "http",
           [F(f"em/cremi/sample_{s}_20160501.hdf", f"https://cremi.org/static/data/sample_{s}_20160501.hdf", h, "--insecure")
            for s, h in (("A", "4c563d1b78acb2bcfb3ea958b6fe1533422f7f4a19f3e05b600bfa11430b510d"),
                         ("B", "887e85521e00deead18c94a21ad71f278d88a5214c7edeed943130a1f4bb48b8"),
                         ("C", "2874496f224d222ebc29d0e4753e8c458093e1d37bc53acd1b69b19ed1ae7052"))],
           ["local_mount:cremi:A", "local_mount:cremi:B", "local_mount:cremi:C"], ["cremi"],
           "only z0-74 of each sample (z75-124 = this project's val/test); cremi.org has an expired TLS certificate -> --insecure, "
           "integrity guaranteed by the published sha256"),
    Source("isbi2012", "ISBI 2012 EM segmentation challenge", "drosophila larva ventral nerve cord, membranes -> cells", "drosophila", "ssTEM",
           "4x4x50", ["cell"], "gold", "challenge terms", "Arganda-Carreras et al., Front. Neuroanat. 2015",
           "https://imagej.net/events/isbi-2012-segmentation-challenge", "http",
           [F("em/isbi2012/isbi.h5", "https://oc.embl.de/index.php/s/h0TkwqxU0PJDdMd/download", "0e10fe909a1243084d91773470856993b7d40126a12e85f0f1345a7a9e512f29")],
           ["downloaded:isbi2012"], ["isbi2012"]),
    Source("vnc", "Drosophila larva VNC (Gerhard et al. 2013)", "20 ssTEM sections, binary membranes -> cells", "drosophila", "ssTEM",
           "4.6x4.6x50", ["cell"], "gold", "CC BY-NC-SA 3.0", "Gerhard et al., figshare 2013, doi:10.6084/m9.figshare.856713",
           "https://github.com/unidesigner/groundtruth-drosophila-vnc", "http",
           [F("em/vnc/vnc-master.zip", "https://github.com/unidesigner/groundtruth-drosophila-vnc/archive/refs/heads/master.zip",
              "f7bd0db03c86b64440a16b60360ad60c0a4411f89e2c021c7ee2c8d6af3d7e86")],
           ["downloaded:vnc"], ["vnc"], "cells = 4-connected components of the non-membrane pixels"),
    Source("wafer4", "Wafer4 (CAD, CVPR 2024)", "mouse medial entorhinal cortex layer 6, dense neurons", "mouse", "multi-beam SEM", "8x8x35",
           ["cell"], "gold", "CC BY-NC 4.0", "Liu et al., CVPR 2024", "https://github.com/liuxy1103/CAD", "http",
           [F("em/wafer4/raw.h5", "https://drive.usercontent.google.com/download?id=1l8Lhk-icIWyDb3fDt_2dDIvvx7S_9vLS&confirm=xxx",
              "5bb64ae54d5d89a501b6942a999a4210eb178671293b5053ad63e12211df603a"),
            F("em/wafer4/labels.h5", "https://drive.usercontent.google.com/download?id=1yyr3eo3-IQsEVvdZIgLsf_QrzaqXfUUX&confirm=xxx",
              "7d3eab21447a0efe327b5192303cc8df8f74ca280eb04e020cbe7dc27beb59c4")],
           ["downloaded:wafer4"], ["wafer4"]),
    Source("fib25", "FIB-25 (FFN release)", "drosophila optic-lobe medulla, proofread neurons", "drosophila", "FIB-SEM", "8x8x8", ["cell"], "gold",
           "see FlyEM", "Takemura et al., PNAS 2015; Januszewski et al., Nat. Methods 2018", "https://github.com/google/ffn", "http",
           [F(f"em/fib25/{n}_{k}.h5", f"{GCS_FIB25}/{n}/{f}", h) for n, k, f, h in (
               ("training_sample2", "raw", "grayscale_maps.h5", "ea031c98ee2de778a9a3a1e6d410df5de73e4ac28022df8e7255d84e3394cafa"),
               ("training_sample2", "gt", "groundtruth.h5", "fd508e7aee1fe51ac9ae0460db4a841d275236f013c1f2552314b4f21b1010ea"),
               ("validation_sample", "raw", "grayscale_maps.h5", "400ccb2a7268a3880c63656e0d794f8e6252e62031869455cc8caeef245b2a83"),
               ("validation_sample", "gt", "groundtruth.h5", "2c5e31af0af5476bc9669b88d01a4570a26eb020799eaf6131aa75f2f7d92e98"),
               ("tstvol-520-1", "raw", "raw.h5", "0667e701c8b4464003d8a6cb0cf9deb2aa79fb415ec51deeac92e5f9c67a5a66"),
               ("tstvol-520-1", "gt", "groundtruth.h5", "ae61ae78a9874eb35ae8e5ed29b4cbfe7bbd07a61789ddb70aef4deb2532eb4e"))],
           ["downloaded:fib25"], ["fib25"], "the same volumes are re-released inside EMNeuron; the dedup stage keeps one copy"),
    Source("emneuron", "EMNeuron (SegNeuron, MICCAI 2024)", "16-source multi-species neuron segmentation collection", "multiple",
           "ATUM-SEM / ssTEM / FIB-SEM / SBF-SEM", "6-9 nm xy, 8-50 nm z", ["cell"], "gold", "MIT (+ licences of the original data)",
           "Zhang et al., MICCAI 2024", "https://huggingface.co/datasets/yanchaoz/EMNeuron", "http",
           [F("em/emneuron/labeled.rar", "https://huggingface.co/datasets/yanchaoz/EMNeuron/resolve/main/labeled.rar"),
            F("em/emneuron/valid.rar", "https://huggingface.co/datasets/yanchaoz/EMNeuron/resolve/main/valid.rar")],
           [f"local_mount:emn:{k}" for k in ("AxonEM[H]-atum", "AxonEM[M]-sstem", "Fib-25-fib", "Hemi-brain-fib", "J0126-sbem", "Mira-adwt",
                                             "Mira-fish", "Mira-scn")] +
           [f"local_mount:emnval:{k}" for k in ("FAFB-1", "FAFB-2", "FIB25-1", "FIB25-2", "Kstr", "Harris", "Ionsem", "Microns[B]")],
           ["emneuron_*"],
           "subsets CREMI-sstem and Kasthuri-atum are NOT converted (they contain this project's CREMI / SNEMI-AC3 benchmark data); "
           "valid/InDistribution/InDistribution/* are byte copies and are skipped; RAR5 -> extracted with libarchive (bsdtar)"),
    Source("kharris15", "SynapseWeb hippocampus CA1 (Harris et al. 2015)", "rat CA1 neuropil, densely reconstructed axons/dendrites/glia",
           "rat", "ssTEM", "~2x2x50", ["cell"], "gold", "CC-BY 4.0 (Open NeuroData)", "Harris et al., Sci. Data 2015",
           "https://neurodata.io/", "stream", [], ["stream:kharris15"], ["kharris15"], "only the 3 densely annotated ROIs (torch_em DENSE_ROIS)"),
    Source("tumor_spheroid", "Tumour spheroid FIB-SEM 2D GT (S-BIAD3263)", "manually annotated cells and nuclei on 2D slices", "human",
           "FIB-SEM", "50 / 100 iso", ["cell", "nucleus"], "gold", "CC BY 4.0", "BioImage Archive S-BIAD3263",
           "https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD3263", "listing",
           [F("em/tumor_spheroid/ground_truths_annotations_filelist.tsv",
              "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/263/S-BIAD3263/Files/ground_truths_annotations_filelist.tsv")],
           ["downloaded:tumor_spheroid"], ["tumor_spheroid"], "file list = the BioStudies TSV; torch_em's /pub/databases path is dead (404)"),
    Source("platynereis", "Platynereis EM training data (Vergara et al. 2021)", "whole-body Platynereis larva: cells (membranes), nuclei, cilia",
           "Platynereis dumerilii", "SBF-SEM", "10x10x25 (approx.)", ["cell", "nucleus", "cilium"], "gold", "CC-BY 4.0",
           "Vergara et al., Cell 2021", f"{ZEN}/record/3675220", "http",
           [F(f"em/platynereis/{n}.zip", f"{ZEN}/record/3675220/files/{n}.zip", h) for n, h in (
               ("membrane", "30eb50c39e7e9883e1cd96e0df689fac37a56abb11e8ed088907c94a5980d6a3"),
               ("nuclei", "a05033c5fbc6a3069479ac6595b0a430070f83f5281f5b5c8913125743cf5510"),
               ("cilia", "6d2b47f63d39a671789c02d8b66cad5e4cf30eb14cdb073da1a52b7defcc5e24"))],
           ["downloaded:platynereis"], ["platynereis_*"]),
    Source("sponge_em", "Sponge EM (Musser et al. 2021)", "freshwater sponge choanocyte chambers: cells, cilia, microvilli", "Spongilla lacustris",
           "FIB-SEM", "unknown", ["cell", "cilium", "microvillus"], "gold", "CC-BY 4.0", "Musser et al., Science 2021", f"{ZEN}/record/8150818", "http",
           [F("em/sponge_em/sponge_em_train_data.zip", f"{ZEN}/record/8150818/files/sponge_em_train_data.zip?download=1",
              "f1df616cd60f81b91d7642933e9edd74dc6c486b2e546186a7c1e54c67dd32a5")],
           ["downloaded:sponge_em"], ["sponge_em"]),
    # ------------------------------------------------------------------ dense cell instances (automatic, silver)
    *[Source(f"stream_{k}", n, c, o, m, r, ["cell"], "silver", "CC-BY 4.0", cit, h, "stream", [], [f"stream:{k}"], [f"stream_{k}"],
             "48 random 1024x1024x8 slabs (seed 0) inside the segmentation bounds shrunk by 10 %; slab kept only if the segmentation "
             "covers >= 50 % of it")
      for k, n, c, o, m, r, cit, h in (
          ("hemibrain", "FlyEM hemibrain v1.2", "drosophila central brain, automatic + proofread segmentation", "drosophila", "FIB-SEM", "8x8x8",
           "Scheffer et al., eLife 2020", "gs://neuroglancer-janelia-flyem-hemibrain (emdata/clahe_yz/jpeg, v1.2/segmentation)"),
          ("manc", "FlyEM MANC v1.2", "drosophila male ventral nerve cord", "drosophila", "FIB-SEM", "8x8x8", "Takemura et al., eLife 2024",
           "gs://flyem-vnc-2-26-.../v3_emdata_clahe_xy/jpeg, gs://manc-seg-v1p2/manc-seg-v1.2"),
          ("malecns", "FlyEM male CNS v0.9", "drosophila male central nervous system", "drosophila", "FIB-SEM", "8x8x8", "FlyEM 2025 (bioRxiv)",
           "gs://flyem-male-cns/em/em-clahe-jpeg, gs://flyem-male-cns/v0.9/segmentation"),
          ("h01", "H01 human cortex (c3 segmentation)", "human temporal cortex", "human", "multi-beam SEM (ATUM)", "8x8x33",
           "Shapson-Coe et al., Science 2024", "gs://h01-release/data/20210601/{4nm_raw (mip1), c3}"),
          ("minnie65", "MICrONS minnie65 (seg m1300)", "mouse visual cortex", "mouse", "ssTEM", "8x8x40", "MICrONS Consortium, Nature 2025",
           "bossdb iarpa_microns/minnie/minnie65/em, gs://iarpa_microns/minnie/minnie65/seg_m1300"),
          ("flywire", "FAFB v14 + FlyWire v783", "adult drosophila whole brain", "drosophila", "ssTEM", "16x16x40", "Dorkenwald et al., Nature 2024",
           "gs://microns-seunglab/drosophila_v0/alignment/image_rechunked (mip2), gs://flywire_v141_m783"))],
    # ------------------------------------------------------------------ organelle / particle instances
    Source("lucchi", "EPFL hippocampus mitochondria (Lucchi et al. 2012)", "mouse CA1 mitochondria, binary -> instances", "mouse", "FIB-SEM",
           "5x5x5", ["mitochondrion"], "gold", "EPFL CVLab (research use)", "Lucchi et al., IEEE TMI 2012",
           "https://www.epfl.ch/labs/cvlab/data/data-em/", "http",
           [F(f"em/lucchi/{n}.tif", f"{EPFL}/{n}.tif") for n in ("training", "training_groundtruth", "testing", "testing_groundtruth")],
           ["local_mount:lucchi"], ["lucchi"], "the testing volume is pixel-identical to Lucchi++ (CEM benchmark) -> resolved by the dedup stage"),
    Source("cem_benchmarks", "CEM mitochondria benchmarks (EMPIAR-10982)", "6 vEM volumes (C. elegans, fly brain, glycolytic muscle, HeLa, "
           "Lucchi++, salivary gland) + 100 TEM images, mitochondria instances", "multiple", "vEM / TEM", "5-24 xy", ["mitochondrion"], "gold",
           "CC0 (EMPIAR)", "Conrad & Narayan, Cell Systems 2023", "https://www.ebi.ac.uk/empiar/EMPIAR-10982/", "listing",
           [F(f"em/cem_benchmarks/{d}_{k}.tif", f"{EBI}/10982/data/mito_benchmarks/{d}/{d}_{k}.tif")
            for d in ("c_elegans", "fly_brain", "glycolytic_muscle", "hela_cell", "lucchi_pp", "salivary_gland") for k in ("em", "mito")] +
           [F("em/cem_benchmarks/tem_benchmark_metadata.csv", f"{EBI}/10982/data/tem_benchmark_metadata.csv")],
           ["downloaded:cem_benchmarks", "downloaded:cem_tem"], ["cem_c_elegans", "cem_fly_brain", "cem_glycolytic_muscle", "cem_hela_cell", "cem_lucchi_pp", "cem_salivary_gland", "cem_tem_benchmark"],
           "TEM benchmark images/masks are listed from the EMPIAR directory index at download time; pixel sizes from the EMPIAR API "
           "(Å/10), z spacing not given -> set to xy"),
    Source("cem_mitolab", "CEM-MitoLab (EMPIAR-11037)", "21,860 annotated 2D patches from many vEM/TEM datasets, mitochondria instances",
           "multiple", "vEM / TEM", "heterogeneous (not given per patch)", ["mitochondrion"], "gold", "CC0 (EMPIAR)",
           "Conrad & Narayan, Cell Systems 2023", "https://www.ebi.ac.uk/empiar/EMPIAR-11037/", "http",
           [F("em/cem/cem_mitolab.zip", f"{EBI}/11037/data/cem_mitolab.zip")], ["downloaded:cem_mitolab"], ["cem_mitolab"],
           "may contain patches of other public datasets (sub-dataset names kept for published data) -> covered by the dedup stage"),
    Source("mitoem", "MitoEM-R / MitoEM-H (Wei et al. 2020)", "rat and human cortex mitochondria, 2 x 500 labelled 4096^2 sections",
           "rat, human", "multi-beam SEM (ATUM)", "8x8x30", ["mitochondrion"], "gold", "CC BY-NC 4.0", "Wei et al., MICCAI 2020",
           "https://mitoem.grand-challenge.org/", "http",
           [F("em/mitoem/EM30-R-im.zip", "https://huggingface.co/datasets/pytc/EM30/resolve/main/EM30-R-im.zip",
              "6a2cac68adde5d01984542d3ee1d7753d1fa3e6eb2a042ce15ce297c95885bbe"),
            F("em/mitoem/EM30-H-im.zip", "https://www.dropbox.com/s/z41qtu4y735j95e/EM30-H-im.zip?dl=1",
              "98fe259f36a7d8d43f99981b7a0ef8cdeba2ce2615ff91595f428ae57207a041"),
            F("em/mitoem/EM30-R-mito-train-val-v2.zip", "https://huggingface.co/datasets/pytc/MitoEM/resolve/main/EM30-R-mito-train-val-v2.zip",
              "c56380ac575428a818bd293ca3509d1249999846c3702ccbf11d308acdd2ae86"),
            F("em/mitoem/EM30-H-mito-train-val-v2.zip", "https://www.dropbox.com/s/dhf89bc14kemw4e/EM30-H-mito-train-val-v2.zip?dl=1",
              "0e8ed292cfcd0c58701d9f4299244a1b66d6aeb506c85754c34f98a4eda0ef1b")],
           ["downloaded:mitoem"], ["mitoem_R", "mitoem_H"], "only labelled sections 0-499 (500-999 have no public labels); "
           "macOS __MACOSX/._* entries in the zips are ignored"),
    Source("mitoemv2", "MitoEM v2 (2025)", "8 tissues / cell types (beta, jurkat, macrophage, mossy fibre, podocyte, pyramidal, sperm, "
           "stem), expert-verified mitochondria", "multiple", "vEM", "varies", ["mitochondrion"], "gold", "CC-BY 4.0", "MitoEM v2, Zenodo 17635006",
           f"{ZEN}/records/17635006", "http",
           [F(f"em/mitoemv2/{n}.zip", f"{ZEN}/records/17635006/files/{n}.zip") for n in (
               "Dataset001_ME2-Beta", "Dataset002_ME2-Jurkat", "Dataset003_ME2-Macro", "Dataset004_ME2-Mossy", "Dataset005_ME2-Podo",
               "Dataset006_ME2-Pyra", "Dataset007_ME2-Sperm", "Dataset008_ME2-Stem")],
           ["downloaded:mitoemv2"], ["mitoemv2_*"], "train + val samples of split.json only; test samples (labelsTs) held out as "
           "an external benchmark"),
    Source("asem", "ASEM mitochondria (Gallusser et al. 2023)", "cultured cells (SUM159, iPSC-derived, beta cells), mitochondria masks",
           "human / mouse cells", "FIB-SEM", "4-5 iso", ["mitochondrion"], "gold", "CC-BY 4.0", "Gallusser et al., J. Cell Biol. 2023",
           "https://open.quiltdata.com/b/asem-project", "stream", [], ["asem:main"], ["asem_mito"],
           "cells 2E, 3A, 46, 58, 61, 64 (those with mito labels); every 4th section kept (4-5 nm isotropic -> 16-20 nm spacing); "
           "ER / Golgi / CCP labels are semantic networks and are not converted"),
    Source("uro_cell", "UroCell (Mekuč et al. 2020)", "mouse urothelium umbrella cells: mitochondria, fusiform vesicles, endolysosomes",
           "mouse", "FIB-SEM", "16x16x15", ["mitochondrion", "fusiform_vesicle", "endolysosome"], "gold", "CC BY 4.0",
           "Mekuč et al., Comput. Biol. Med. 2020", "https://github.com/MancaZerovnikMekuc/UroCell", "http",
           [F("em/uro_cell/UroCell-master.zip", "https://github.com/MancaZerovnikMekuc/UroCell/archive/refs/heads/master.zip",
              "a48cf31b06114d7def642742b4fcbe76103483c069122abe10f377d71a1acabc")],
           ["downloaded:uro_cell"], ["uro_cell"], "one multi-class file per sub-volume; Golgi (approximate regions) not used"),
    Source("nuclei", "FAFB / MICrONS nuclei crops (torch_em)", "nucleus instances from whole-connectome nucleus segmentations",
           "drosophila, mouse", "ssTEM (downsampled)", "not given", ["nucleus"], "silver", "see original publications",
           "Mu et al. 2021 (FAFB nuclei); MICrONS Consortium 2025", "https://github.com/constantinpape/torch-em", "http",
           [F("em/fafb_nuclei/fafb_nuclei.zip", "https://owncloud.gwdg.de/index.php/s/PDrkO02w7FWLrsh/download",
              "7f245866fa8ffb473f7bda1fbbea23d265183e4ada5e34a36ecd60475809dca3"),
            F("em/microns_nuclei/microns_nuclei.zip", "https://owncloud.gwdg.de/index.php/s/ToLGAzg1FAV4Sxf/download",
              "36afcc963aea597faf991f6844537d2330739a89aa05c1a91fea31f2b4dc2de4")],
           ["downloaded:nuclei"], ["fafb_nuclei", "microns_nuclei"]),
    Source("nucmm_z", "NucMM-Z (Lin et al. 2021)", "zebrafish whole-brain nuclei", "zebrafish", "SBF-SEM", "~0.51x0.51x0.48 um (downsampled)",
           ["nucleus"], "gold", "CC-BY 4.0", "Lin et al., MICCAI 2021", "https://drive.google.com/drive/folders/1_4CrlYvzx0ITnGlJOHdgcTRgeSkm9wT8",
           "gdown", [F("em/nuc_mm", "https://drive.google.com/drive/folders/1_4CrlYvzx0ITnGlJOHdgcTRgeSkm9wT8")], ["downloaded:nucmm_z"], ["nucmm_z"],
           "only the zebrafish (EM) part; NucMM-M is micro-CT (not EM) and is not used. PARTIAL: Google Drive quota -> images "
           "54/54 blocks but labels only 18/54 (train 18/27, val 0/27); only labelled blocks are converted"),
    Source("probtem", "Mitochondria TEM (Yoon lab, ProbTEM)", "TEM images of mitochondria (skeletal muscle etc.)", "multiple", "TEM", "unknown",
           ["mitochondrion"], "gold", "see source", "Yoon lab, UNIST", "https://yoonlab.unist.ac.kr/index.php/research/mitochondria-tem-dataset/",
           "gdown", [F("em/probtem", "https://drive.google.com/drive/folders/1n2ZqbJEHPyMB_6a6OTBBACt5Jct2PZJc")], ["downloaded:probtem"], ["probtem"],
           "EXCLUDED from the default corpus: Google Drive quota -> only 6/21 train images "
           "(no train masks) and 6/6 test images + masks obtained, 1 mask size-mismatched -> 5 usable images; with the "
           "sqrt-mass weight these 5 images would be heavily over-sampled (0.8 % mass). Re-enable when the full set downloads.",
           default=False),
    Source("aimseg", "AimSeg (2023)", "mouse corpus callosum TEM: axons and myelin", "mouse", "TEM", "unknown", ["axon", "myelin"], "gold",
           "CC-BY 4.0", "Vega-Chica et al. 2023, Zenodo 8351731", f"{ZEN}/records/8351731", "http",
           [F("em/aimseg/Control_Dataset.rar", f"{ZEN}/records/8351731/files/Control_Dataset.rar"),
            F("em/aimseg/Validation_Dataset_v1.rar", f"{ZEN}/records/8351731/files/Validation_Dataset_v1.rar")],
           ["downloaded:aimseg"], ["aimseg"], "axon instances are only the myelinated axons -> own class 'axon' (not the exhaustive 'cell' tiling); "
           "inner-tongue instances are background of both classes"),
    Source("emps", "EMPS nanoparticles (Yildirim & Cole 2021)", "465 EM images of nanoparticles from publications, particle instances",
           "none (materials)", "TEM / SEM", "unknown", ["nanoparticle"], "gold", "MIT", "Yildirim & Cole, J. Chem. Inf. Model. 2021",
           "https://github.com/by256/emps", "http", [F("em/emps/emps-main.zip", "https://github.com/by256/emps/archive/refs/heads/main.zip")],
           ["downloaded:emps"], ["emps"], "non-biological EM; kept for shape / boundary diversity (low sampling mass by sqrt rule)"),
    # ------------------------------------------------------------------ natural images
    Source("coco_panoptic", "COCO panoptic 2017", "natural scenes; every panoptic segment (things + stuff) as an instance", "-", "RGB photo -> gray",
           "-", ["natural_segment"], "gold", "CC-BY 4.0 (annotations), Flickr terms (images)", "Kirillov et al., CVPR 2019; Lin et al., ECCV 2014",
           "https://cocodataset.org/#panoptic-2020", "http",
           [F("natural/coco/panoptic_annotations_trainval2017.zip", "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"),
            F("natural/coco/val2017.zip", "http://images.cocodataset.org/zips/val2017.zip"),
            F("natural/coco/train2017.zip", "http://images.cocodataset.org/zips/train2017.zip")],
           ["coco:val", "coco:train"], ["coco_panoptic"], "used only in em_plus_natural (5 % of the sampling mass)"),
    # ------------------------------------------------------------------ in-house (not public, not in the default manifests)
    Source("mouse30um_local", "mouse_30um mip1 (in-house)", "mouse cortex, dense neurons", "mouse", "ATUM-SEM (in-house)", "16x16x30", ["cell"],
           "gold", "in-house, not for redistribution", "-", "in-house (EMDATA_LOCAL_MOUNT)", "local", [], ["local_mount:mouse30um"],
           ["mouse30um_local"], "only z0-69 (z70-99 = this project's val/test); written to local/, excluded from the manifests", default=False),
]

BY_ID = {s.id: s for s in SOURCES}

# Redistribution tier of each source (decides which HF configs a source may enter; from the licence text of
# the release page). open = CC0 / CC-BY / MIT with attribution -> publishable on the HF Hub;  nc = non-commercial licence
# (CC BY-NC[-SA]) -> only in a separate NC config;  check = licence of the underlying data not verified per item
# (collections that inherit the original licences, "see publication");  restricted = challenge / research-use terms,
# image rights held by third parties (COCO / Flickr), in-house data -> local use only, never uploaded.
REDISTRIBUTION = {
    "open": ["cremi", "kharris15", "tumor_spheroid", "platynereis", "sponge_em", "stream_hemibrain", "stream_manc", "stream_malecns",
             "stream_h01", "stream_minnie65", "stream_flywire", "cem_benchmarks", "cem_mitolab", "mitoemv2", "asem", "uro_cell",
             "nucmm_z", "aimseg", "emps"],
    "nc": ["vnc", "wafer4", "mitoem"],
    "check": ["emneuron", "fib25", "nuclei", "probtem"],
    "restricted": ["snemi3d", "isbi2012", "lucchi", "coco_panoptic", "mouse30um_local"],
}
for _tier, _ids in REDISTRIBUTION.items():
    for _i in _ids:
        object.__setattr__(BY_ID[_i], "redistribution", _tier)
assert all(s.id in sum(REDISTRIBUTION.values(), []) for s in SOURCES), "every source needs a redistribution tier"


def host_policy(url):
    """(connections per file, parallel files) for a URL"""
    if "zenodo.org" in url:
        return 1, 1
    if "ftp.ebi.ac.uk" in url:
        return 2, 2
    return 6, 4


def owner_of(dataset_name):
    """registry id that produced an output dataset name"""
    for s in SOURCES:
        for o in s.outputs:
            if dataset_name == o or (o.endswith("*") and dataset_name.startswith(o[:-1])):
                return s
    return None
