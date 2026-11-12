import glob
import os
import subprocess
import time

from collections import defaultdict
from importlib import metadata
from typing import TYPE_CHECKING, Optional

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# --- Siemens imports
from shs_acit_concord.concord import concord_process
from shs_acit_prague_atlas_loader.load_atlas import Atlas as PragueAtlas
from shs_acit_quant_periles.ring_analyzer import RingAnalyzer
from shs_acit_quant_perivent.compute_qval_in_perivent import Perivent_zscores
from shs_acit_quant_regions.compute_qval_in_regions import ZscoresMBRegions
from shs_acit_quant_wmtracts.compute_qval_in_wmtracts import WMLesions_zscores
from shs_acit_registration.registration import run_elastix, run_transformix
from shs_acit_utils.general import logger_h, ImageReorienterApp_sitk
from shs_acit_wmh_phoenix.phoenix import Phoenix
from shs_acit_t2overt1_atlas_loader.load_atlas import Atlas as AtlasT2overT1
from shs_acit_t2overt1.compute_t2overt1_zscores import T2OverT1ZScore
from shs_acit_zscores.zscores_utils import ZScore  # 2026-05-26-UPDATE

if TYPE_CHECKING:
    import logging


def get_version(package: str) -> str:
    """This function returns the version of a package
    Args:
        package: name of the package
    Returns:
        version_str: version of the package
    """
    try:
        version_str = metadata.version(package).replace(".", "_")
    except metadata.PackageNotFoundError:
        version_str = "not_found"
    return version_str


VERSION_DL_MORPHOBOX = "1.2.7"
PATH_DL_MORPHO_EXE = r"executables\shs_dl_morphometry.exe"

# (label, mapa v MP2RAGE prostoru, mapa v SST, whole-brain mapa v SST)
QMAPS = [
    ("t1", "t1_zscores", "t1_zscores_sst", "t1_zscores_wholebrain"),
    ("t2", "t2_zscores", "t2_zscores_sst", "t2_zscores_wholebrain"),
    ("t2overt1", "t2overt1_zscores", "t2overt1_zscores_sst", "t2overt1_zscores_wholebrain"),
]


def log_execution_time(func):
    """Decorator to wrap a function with logging for start and finish.

    Parameters
    ----------
    func: method
        The method to be wrapped. Has to be method of class Processing, because it uses self.logger
        and self.running
    """

    def wrapper(self, *args, **kwargs):
        self.running = f"[PROGRAM] - {func.__name__.upper().removeprefix('RUN_')}"
        self.logger.info("%s - Starting", self.running)

        start_time = time.time()
        result = func(self, *args, **kwargs)
        run_time = time.time() - start_time

        self.logger.info("%s - Finished in %.2f seconds", self.running, run_time)
        return result

    return wrapper


class Processing:
    """Zpracování jednoho vyšetření. Třída je stejná pro všechna centra,
    rozdíly (názvy souborů, sub/ses struktura) se předávají z main().

    Parameters
    ----------
    session: None pro data bez ses- složek (jen sub-XXX)
    file_dict: glob vzory vstupních souborů pro dané centrum
    anat_subdir: True, pokud jsou soubory ve složce anat/
    uni_priority: pořadí preferovaných přípon UNI souboru, pokud jich je víc (Praha)
    allow_multiple: při více shodách vzít první abecedně (jinak None)
    force: kroky, které se přepočítají i když výstupy existují, např. {"periventricular"}
    """

    def __init__(
            self,
            subject: str,
            session: Optional[str],
            data_path: str,
            derivatives_path: str,
            age: int,
            sex: str,
            logger: "logging.Logger",
            file_dict: dict,
            anat_subdir: bool = True,
            uni_priority: Optional[list] = None,
            allow_multiple: bool = False,
            force: Optional[set] = None,
    ):
        self.subject = subject
        self.session = session
        self.data_path = data_path

        self.derivatives_path = derivatives_path
        os.makedirs(self.derivatives_path, exist_ok=True)

        self.patient_age = age
        self.patient_sex = sex
        parts = [subject] + ([session] if session else [])
        self.output_path = os.path.join(derivatives_path, "{}", *parts)
        self.session_path = os.path.join(self.data_path, *parts, *(["anat"] if anat_subdir else []))
        self.logger = logger
        self.running = ""
        self.force = set(force or ())
        self.logger.info(f"STARTING PROCESSING OF SUBJECT {subject}, SESSION {session}")

        # defaultdict(None) se chová jako dict (KeyError) -> chybějící klíč vrátí None
        self.files = defaultdict(lambda: None)
        for key, value in file_dict.items():
            self.files[key] = self.find_files(
                os.path.join(self.session_path, value),
                priority=uni_priority if key == "uni" else None,
                allow_multiple=allow_multiple,
            )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def find_files(glob_string: str, priority: Optional[list] = None, allow_multiple: bool = False):
        output = sorted(glob.glob(glob_string))
        if not output:
            return None
        for suffix in priority or []:
            matches = [p for p in output if p.endswith(suffix)]
            if matches:
                return matches[0]
        if len(output) == 1 or allow_multiple:
            return output[0]
        return None

    @staticmethod
    def _first(pattern: str) -> Optional[str]:
        """First file matching the glob pattern, or None."""
        matches = sorted(glob.glob(pattern, recursive=True))
        return matches[0] if matches else None

    def _done(self, step: str, pattern: str) -> Optional[str]:
        """Existing output of a step (None when missing or when the step is forced)."""
        if step in self.force:
            return None
        return self._first(pattern)

    def _require(self, *keys: str) -> bool:
        """Check that all inputs of a step are available; otherwise log and skip the step."""
        missing = [k for k in keys if not self.files.get(k)]
        if missing:
            self.logger.warning("%s - skipping, missing inputs: %s", self.running, missing)
            return False
        return True

    # ------------------------------------------------------------------ steps
    def _zscore_outputs(self, output_path: str, lib_output: Optional[dict]) -> dict:
        """Cesty k výstupům z-score pro TENTO subjekt/session.

        Cesta z voxelwise_zscores.output se použije jen pokud leží v output_path – objekt je
        sdílený pro všechny subjekty, takže po FileExistsError může ukazovat na předchozí subjekt.
        Jinak se soubor dohledá ve složce output_path (podsložka podle klíče z .output).
        """
        lib_output = lib_output or {}
        root = os.path.abspath(output_path)

        def from_lib(out_key: str, fname: str) -> Optional[str]:
            if lib_output.get(out_key):
                cand = os.path.abspath(os.path.join(lib_output[out_key], fname))
                if cand.startswith(root + os.sep):
                    return cand
            return self._first(os.path.join(output_path, "**", f"*{out_key}*", fname))

        outputs = {
            "t1_zscores_sst": from_lib("transformix_T1_template", "result.nii.gz"),
            "t2_zscores_sst": from_lib("transformix_T2_template", "result.nii.gz"),
            "t2overt1_zscores_sst": from_lib("transformix_T2OverT1_template", "T2OverT1map_sst.nii.gz"),
            "t1_zscores": os.path.join(output_path, "T1ZscoreMap_SC.nii.gz"),
            "t2_zscores": os.path.join(output_path, "T2ZscoreMap_SC.nii.gz"),
            "t2overt1_zscores": os.path.join(output_path, "T2OverT1ZscoreMap_SC.nii.gz"),
            "prior_in_mp2rage": os.path.join(output_path, "T1_prior_SC.nii.gz"),
            "tp_uni_to_sst": from_lib("elastix_uni_template", "TransformParameters.1.txt"),
            "t1_zscores_wholebrain": from_lib("zscores", "T1_zscores_wholebrain_sst.nii.gz"),
            "t2_zscores_wholebrain": from_lib("zscores", "T2_zscores_wholebrain_sst.nii.gz"),
            "t2overt1_zscores_wholebrain": from_lib("zscores", "T2OverT1_zscores_wholebrain_sst.nii.gz"),
            "reoriented_inv2": from_lib("reorient", "inv2_reoriented.nii.gz"),
        }
        return {k: (v if v and os.path.isfile(v) else None) for k, v in outputs.items()}

    @log_execution_time
    def run_zscores(self, voxelwise_zscores: "ZScore"):
        if self.files["t2map"] is None:
            self.logger.info(f"[Z score] T2map file not found {self.session_path}")

        if any(self.files[x] is None for x in ["inv2", "t1map", "uni"]):
            self.logger.error(f"[Z score] MP2RAGE file not found {self.session_path}")
            return

        version = metadata.version("shs_acit_zscores")
        output_path = self.output_path.format(f"zscores_v{version}")
        os.makedirs(output_path, exist_ok=True)

        required = ["t1_zscores_sst", "t1_zscores", "prior_in_mp2rage", "tp_uni_to_sst"]
        if self.files["t2map"] is not None:
            required += ["t2overt1_zscores_sst", "t2overt1_zscores"]

        # 1) Už spočítáno? -> jen naplnit self.files, nic nespouštět
        outputs = self._zscore_outputs(output_path, None)
        if "zscores" not in self.force and all(outputs[k] for k in required):
            self.logger.info("%s - outputs exist, skipping computation", self.running)
            self.files.update(outputs)
            return

        # 2) Výpočet
        try:
            voxelwise_zscores(
                age=self.patient_age,
                sex=self.patient_sex,
                inv2=self.files["inv2"],
                uni=self.files["uni"],
                t1map=self.files["t1map"],
                t2map=self.files["t2map"],
                outdir=output_path,
                use_dl_registration=False,
                use_native_space=False,
            )
        except FileExistsError as err:
            self.logger.warning("%s -- %s -- skipping", self.running, err)
        except Exception as err:
            self.logger.error("%s -- %s", self.running, err)
            raise Exception(f"Error occurred: {err}") from err

        # 2026-05-26-UPDATE: bez T2 mapy ZScore ukládá mapy bez přípony _SC
        if self.files["t2map"] is None:
            for src, dst in [("T1ZscoreMap.nii.gz", "T1ZscoreMap_SC.nii.gz"),
                             ("T1prior.nii.gz", "T1_prior_SC.nii.gz")]:
                src_p, dst_p = os.path.join(output_path, src), os.path.join(output_path, dst)
                if os.path.isfile(src_p) and not os.path.isfile(dst_p):
                    os.rename(src_p, dst_p)

        # 3) Naplnit self.files
        outputs = self._zscore_outputs(output_path, getattr(voxelwise_zscores, "output", None))
        self.files.update(outputs)
        missing = [k for k in required if not outputs[k]]
        if missing:
            self.logger.error("%s - outputs missing after computation: %s", self.running, missing)

    @log_execution_time
    def run_morphobox(self):
        output_path = self.output_path.format(f"morphobox_v{VERSION_DL_MORPHOBOX}")
        os.makedirs(output_path, exist_ok=True)

        def label_files():
            return [f for f in os.listdir(output_path) if f.startswith("dl_label")]

        if (not label_files() or "morphobox" in self.force) and self._require("uni"):
            cmd_list = [
                PATH_DL_MORPHO_EXE,
                "--input_path", f"{self.files['uni']}",
                "--out_dir", output_path,
                "--volume_type", "mp2rage",
                "--magnetic_field_strength", "3",
                "--mac_address", "04:7C:16:5D:B0:19",
                "--patient_age", f"{self.patient_age}",
                "--patient_sex", f"{self.patient_sex}",
                "--compute_zscores", "False",
                "--save_skull_stripping_outputs", "True",
                "--save_outputs_to_disk", "True",
                "--save_zscore_maps", "False",
                "--save_dev_map", "False",
                "--save_regional_T1_dev_map", "False",
                "--force_cpu", "True",
            ]
            result = subprocess.run(
                cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
            )
            if result.returncode == 0:
                out_path_log = os.path.join(output_path, "morphobox_log.txt")
                with open(out_path_log, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
            else:
                self.logger.error(f"\nERROR: {result.stderr}")

        files = label_files()
        if len(files) == 1:
            self.files["MB_label_map"] = os.path.join(output_path, files[0])
        else:
            self.files["MB_label_map"] = None
            self.logger.error(
                "%s - expected exactly one 'dl_label*' file in %s, found %d",
                self.running, output_path, len(files),
            )

    @log_execution_time
    def run_concord(self):
        version = metadata.version("shs_acit_concord")
        output_path = self.output_path.format(f"concord_v{version}")
        os.makedirs(output_path, exist_ok=True)
        pattern = os.path.join(output_path, "*MP2RAGE_summary*")

        if not self._done("concord", pattern) and self._require("inv2", "uni"):
            concord_process([self.files["inv2"], self.files["uni"]], output_path)

        self.files["concord_summary"] = self._first(pattern)

    @log_execution_time
    def run_registration_flair_to_mp2rage(self):
        output_path = self.output_path.format("registration_flair_to_MP2RAGE_v1.0")
        transform_parameter_file = os.path.join(output_path, "TransformParameters.0.txt")

        if not self._done("registration_flair_to_mp2rage", transform_parameter_file) \
                and self._require("flair", "inv2"):
            os.makedirs(output_path, exist_ok=True)
            run_elastix(
                moving=self.files["flair"],
                fixed=self.files["inv2"],
                output_path=output_path,
                param_file_list=["affine"],
                output_filename="flair_reg_to_mp2rage.nii.gz",
                save=True,
            )
        self.files["transform_parameter_file_flair_to_mp2rage"] = (
            transform_parameter_file if os.path.isfile(transform_parameter_file) else None
        )

    @log_execution_time
    def run_registration_lesion_map_to_mp2rage(self):
        output_path = self.output_path.format("registration_lesion_to_MP2RAGE_v1.0")
        output_file = os.path.join(output_path, "lesion_map_reg_to_mp2rage.nii.gz")

        if not self._done("registration_lesion_map_to_mp2rage", output_file) \
                and self._require("lesion_map", "transform_parameter_file_flair_to_mp2rage"):
            os.makedirs(output_path, exist_ok=True)
            run_transformix(
                moving=self.files["lesion_map"],
                transform_param_file=self.files["transform_parameter_file_flair_to_mp2rage"],
                output_path=output_path,
                output_filename="lesion_map_reg_to_mp2rage.nii.gz",
                param_to_modify={
                    "FinalBSplineInterpolationOrder": (None, "0")
                },  # nearest neighbour interpolation (lesion mask is binary)
                save=True,
            )

        self.files["lesion_map_in_MP2RAGE"] = output_file if os.path.isfile(output_file) else None

    def run_registration_lesion_map_to_sst(self):
        output_path = self.output_path.format("registration_lesion_map_to_SST")
        output_file = os.path.join(output_path, "lesion_map_reg_to_sst.nii.gz")
        if not os.path.isfile(output_file):
            os.makedirs(output_path, exist_ok=True)
            # ---- Update 2026.11.11 ----
            lesion_map_in_mp2rage = sitk.ReadImage(self.files["lesion_map_in_MP2RAGE"])
            lesion_map_in_mp2rage_reor, _ = ImageReorienterApp_sitk(lesion_map_in_mp2rage)
            sitk.WriteImage(lesion_map_in_mp2rage_reor,
                            os.path.join(output_path, "lesion_map_in_MP2RAGE_reoriented.nii.gz"))
            self.files["lesion_map_in_MP2RAGE_reoriented"] = os.path.join(output_path,
                                                                          'lesion_map_in_MP2RAGE_reoriented.nii.gz')
            run_transformix(
                moving=self.files["lesion_map_in_MP2RAGE_reoriented"],  # ---- Update 2026.11.11 ----
                transform_param_file=self.files["tp_uni_to_sst"],
                output_path=output_path,
                output_filename="lesion_map_reg_to_sst.nii.gz",
                param_to_modify={
                    "FinalBSplineInterpolationOrder": (None, "0")
                },  # nearest neighbour interpolation (lesion mask is binary)
                save=True,
            )
        self.files["lesion_map_in_sst"] = (

            output_file if os.path.isfile(output_file) else None

        )


    @log_execution_time
    def run_wmh(self, phoenix_wmh: Phoenix):
        version = metadata.version("shs_acit_wmh_phoenix")
        output_path = self.output_path.format(f"wmh_phoenix_v{version}")
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "mr_wmh_phoenix_lesion_map.nii.gz")

        if not self._done("wmh", output_file) and self._require("flair"):
            flair = sitk.ReadImage(self.files["flair"])
            results = phoenix_wmh.process_case(current=flair, return_cross=True)
            sitk.WriteImage(results["wmh_cross"], output_file)

        self.files["lesion_map"] = output_file if os.path.isfile(output_file) else None

    @log_execution_time
    def run_perilesional_rings(self):
        version = metadata.version("shs_acit_quant_periles")
        dist_stats = {}

        for label, qmap_key, _, _ in QMAPS:
            if self.files[qmap_key] is None:
                continue
            output_path = self.output_path.format(f"quant_periles_{label}_v{version}")
            pattern = os.path.join(output_path, "*wm_sc_mask.nii.gz")

            if not self._done("perilesional_rings", pattern) \
                    and self._require("lesion_map_in_MP2RAGE", "prior_in_mp2rage"):
                os.makedirs(output_path, exist_ok=True)
                ra = RingAnalyzer(
                    output_path=output_path,
                    lesions_path=self.files["lesion_map_in_MP2RAGE"],
                    qmap_path=self.files[qmap_key],
                    wm_sc_mask_path=self.files["prior_in_mp2rage"],
                )
                dist_stats[label] = ra.compute_distance_statistics(label="my label")

            self.files[f"periles_{label}_dir"] = output_path if self._first(pattern) else None

        return dist_stats.get("t1"), dist_stats.get("t2"), dist_stats.get("t2overt1")

    @log_execution_time
    def run_periventricular(self, perivent_zscores: Perivent_zscores):
        pkg_version = metadata.version("shs_acit_quant_perivent")

        for label, _, qmap_sst_key, _ in QMAPS:
            if self.files[qmap_sst_key] is None:
                continue
            output_path = self.output_path.format(f"quant_perivent_{label}_v{pkg_version}")
            pattern = os.path.join(output_path, "*qvalues_in_periventricular_bands.csv")

            if not self._done("periventricular", pattern) and self._require("lesion_map_in_sst"):
                os.makedirs(output_path, exist_ok=True)
                perivent_zscores.compute_qvalues_in_perivent(
                    self.files[qmap_sst_key],
                    self.files["lesion_map_in_sst"],
                    output_path=output_path,
                )

            self.files[f"perivent_{label}_csv"] = self._first(pattern)

    @log_execution_time
    def run_quant_regions(self, zscores_stats_in_mb_regions: ZscoresMBRegions):
        pkg_version = metadata.version("shs_acit_quant_regions")

        for label, qmap_key, _, _ in QMAPS:
            if self.files[qmap_key] is None:
                continue
            output_path = self.output_path.format(f"quant_regions_{label}_v{pkg_version}")
            pattern = os.path.join(output_path, "*qvalues_in_regions_NAWM.csv")

            if not self._done("quant_regions", pattern) \
                    and self._require("MB_label_map", "lesion_map_in_MP2RAGE", "prior_in_mp2rage"):
                os.makedirs(output_path, exist_ok=True)
                zscores_stats_in_mb_regions.compute_qvalues_in_regions(
                    qmap_path=self.files[qmap_key],
                    label_path=self.files["MB_label_map"],
                    lesion_mask_path=self.files["lesion_map_in_MP2RAGE"],
                    wm_sc_mask_path=self.files["prior_in_mp2rage"],
                    output_path=output_path,
                )

            self.files[f"regions_{label}_csv"] = self._first(pattern)

    @log_execution_time
    def run_tracts(self, wm_lesions_zscores: WMLesions_zscores):
        pkg_version = metadata.version("shs_acit_quant_wmtracts")

        for label, _, _, qmap_wb_key in QMAPS:
            if self.files[qmap_wb_key] is None:
                continue
            output_path = self.output_path.format(f"quant_wmtracts_{label}_v{pkg_version}")
            pattern = os.path.join(output_path, "*qvalues_in_wm_tracts.csv")

            if not self._done("tracts", pattern) and self._require("lesion_map_in_sst"):
                os.makedirs(output_path, exist_ok=True)
                wm_lesions_zscores.compute_qvalues_in_tracts(
                    self.files[qmap_wb_key],
                    self.files["lesion_map_in_sst"],
                    output_path=output_path,
                )

            self.files[f"tracts_{label}_csv"] = self._first(pattern)


class PatientNameSex:
    """Class used to extract patient age and sex from a demographics table. The class associates
    the subject (and session, if the data have ses- folders) to the corresponding age and sex.

    Parameters
    ----------
    table_path : str
        Path to the Excel file containing the demographics table.
        Should have the following columns: PatientID, AcquisitionDate, PatientSex, PatientAge.
    """

    def __init__(self, table_path: str):
        table = pd.read_excel(table_path)
        self.preprocessing_table = table[
            ["PatientID", "AcquisitionDate", "PatientSex", "PatientAge"]
        ].drop_duplicates()

    @staticmethod
    def _norm(value) -> str:
        """Text bez mezer, koncovky .0 a úvodních nul (Excel často načte ID jako číslo)."""
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        return text.lstrip("0") or "0"

    def __call__(self, subject: str, session: Optional[str] = None):
        t = self.preprocessing_table
        mask = t["PatientID"].map(self._norm) == self._norm(subject[4:])
        if session is not None:  # struktura sub/ses -> párovat i datum vyšetření
            mask &= t["AcquisitionDate"].map(self._norm) == self._norm(session[4:])
        rows = t[mask]

        if rows.empty:
            return None, None

        return int(rows["PatientAge"].values[0]), rows["PatientSex"].values[0]


def get_relax_map_type(prague_atlas: PragueAtlas, relax_map_type: str):
    """Get the T1 or T2 atlas from Prague atlas.

    Parameters
    ----------
    prague_atlas: PragueAtlas
        The Prague atlas object containing the T1 and T2 maps.
    relax_map_type: str
        The type of relaxation map to retrieve, either "t1" or "t2".

    Returns
    -------
    atlas: dict
        A dictionary containing the specified relaxation map.
    """

    relax_map_type = relax_map_type.lower()
    assert relax_map_type in [
        "t1",
        "t2",
        "t1overt2",
    ], "Relaxation map type has to be either t1, t2, or t1overt2"

    atlas = {
        key.removeprefix(f"{relax_map_type}_"): value
        for key, value in prague_atlas.__dict__.items()
        if key.startswith(f"{relax_map_type}_")
    }
    atlas["prior"] = atlas.pop("wm_prior") / 1000

    if "min_age" not in atlas:
        atlas["min_age"] = 18
    if "max_age" not in atlas:
        atlas["max_age"] = 90

    return atlas


def main():  # pylint: disable=missing-function-docstring
    # ============================== NASTAVENÍ CENTRA ==============================
    # Datum NEMĚNIT mezi běhy – je součástí cesty k výstupům, nové datum = vše se počítá znovu
    today = "2026_10_12"
    bids_dir = r"E:\Project_DCE\5_Amsterdam\bids"
    demographics_file = "demographics.xlsx"
    has_sessions = True  # True: sub-XXX/ses-YYY, False: jen sub-XXX (demografie se páruje jen podle ID)
    anat_subdir = True  # soubory ve složce .../anat/
    file_dict = {
        "inv2": "*INV2.nii*.gz",
        "uni": "*UNI_Images.nii*.gz",
        "t1map": "*pipeline_T1map.nii*.gz",
        "t2map": "*t2map*.nii",
        "mprage": "*t1_mprage*.nii.gz",
        "flair": "*3D_FLAIR.nii*.gz",
    }
    uni_priority = None  # preferované UNI soubory, pokud jich je víc
    allow_multiple = False  # více shod -> vzít první abecedně (False = soubor se nepoužije)
    t1_zscore_without_t2map = True  # bez T2 mapy použít ZScore (jen T1), jinak vždy T2OverT1ZScore
    # Kroky, které se mají přepočítat i když výstupy existují (názvy metod bez "run_"),
    # např. po opravě skriptu pro periventrikulární pásy: {"periventricular"}
    FORCE = set()
    # ==============================================================================

    derivatives_path = os.path.join(bids_dir, "derivatives", today)
    os.makedirs(derivatives_path, exist_ok=True)
    demographics = PatientNameSex(
        glob.glob(os.path.join(glob.escape(bids_dir), demographics_file))[0]
    )

    logger = logger_h(
        path_to_logs=os.path.join(derivatives_path, "logs"),
        name="PraguePipeline",
        filename="PraguePipeline",
        verbose=False,
    )

    all_files = {}

    prague_atlas = PragueAtlas()
    prague_atlas_t1overt2 = AtlasT2overT1()

    atlas_t1 = get_relax_map_type(prague_atlas, "t1")
    atlas_t2 = get_relax_map_type(prague_atlas, "t2")

    atlas_t2overt1 = prague_atlas_t1overt2.__dict__
    # /!\ For t2 over t1 we use the T1 white matter prior
    atlas_t2overt1.update({"prior": atlas_t1.get("prior", None)})

    voxelwise_t2overt1_zscores = T2OverT1ZScore(
        atlas_t2overt1=atlas_t2overt1,
        atlas_t1=atlas_t1,
        atlas_t2=atlas_t2,
        logger=logger,
        writetmp=True,
    )
    # 2026-05-26-UPDATE
    voxelwise_t1_zscores = (
        ZScore(atlas_t1=atlas_t1, logger=logger, writetmp=True) if t1_zscore_without_t2map else None
    )
    phoenix_wmh = Phoenix()
    wm_lesions_zscores = WMLesions_zscores(prague_atlas, threshold_value=2.0)
    perivent_zscores = Perivent_zscores(prague_atlas, threshold_value=2.0)
    zscores_stats_in_mb_regions = ZscoresMBRegions(logger=logger, threshold_value=2.0)

    subjects = [
        item
        for item in os.listdir(bids_dir)
        if os.path.isdir(os.path.join(bids_dir, item)) and item.startswith("sub-")
    ]
    for subject in tqdm(subjects):
        if has_sessions:
            sessions = [
                item
                for item in os.listdir(os.path.join(bids_dir, subject))
                if os.path.isdir(os.path.join(bids_dir, subject, item))
                   and item.startswith("ses-")
            ]
        else:
            sessions = [None]  # data jen ve struktuře sub-XXX

        for session in sessions:
            patient_age, patient_sex = demographics(subject, session)
            if patient_age is None:
                logger.warning(
                    "[PROGRAM] - DEMOGRAPHICS - Patient not found in demographics table: %s %s — skipping",
                    subject,
                    session,
                )
                continue

            proc = Processing(
                subject=subject,
                session=session,
                data_path=bids_dir,
                derivatives_path=derivatives_path,
                sex=patient_sex,
                age=patient_age,
                logger=logger,
                file_dict=file_dict,
                anat_subdir=anat_subdir,
                uni_priority=uni_priority,
                allow_multiple=allow_multiple,
                force=FORCE,
            )

            try:
                # Všechny kroky běží vždy: hotové se jen načtou do proc.files, chybějící se dopočítají.
                if voxelwise_t1_zscores is not None and proc.files["t2map"] is None:
                    proc.run_zscores(voxelwise_zscores=voxelwise_t1_zscores)
                else:
                    proc.run_zscores(voxelwise_zscores=voxelwise_t2overt1_zscores)
                proc.run_morphobox()
                proc.run_wmh(phoenix_wmh=phoenix_wmh)
                proc.run_registration_flair_to_mp2rage()
                proc.run_registration_lesion_map_to_mp2rage()
                proc.run_registration_lesion_map_to_sst()

                proc.run_periventricular(perivent_zscores=perivent_zscores)
                proc.run_quant_regions(zscores_stats_in_mb_regions=zscores_stats_in_mb_regions)
                proc.run_perilesional_rings()
                proc.run_tracts(wm_lesions_zscores=wm_lesions_zscores)
                proc.run_concord()

            except Exception as ex:
                logger.critical(
                    "%s -- Critical exception of type %s for %s, %s: %s",
                    proc.running,
                    type(ex),
                    subject,
                    session,
                    ex,
                )

            all_files[(subject, session)] = proc.files


if __name__ == "__main__":
    main()
