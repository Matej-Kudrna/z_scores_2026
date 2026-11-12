import glob
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from importlib import metadata
from typing import TYPE_CHECKING

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
from shs_acit_utils.general import logger_h
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
    def __init__(
        self,
        subject: str,
        session: str,
        data_path: str,
        derivatives_path: str,
        age: int,
        sex: str,
        logger: "logging.Logger",
    ):
        # with the __init__ when you call it, you will define variables, that than be available
        # in all functions in class (if you specify init in argument)
        self.subject = subject
        self.session = session
        self.data_path = data_path

        self.derivatives_path = derivatives_path
        os.makedirs(self.derivatives_path, exist_ok=True)

        self.patient_age = age
        self.patient_sex = sex
        self.output_path = os.path.join(derivatives_path, "{}", subject, session)
        self.session_path = os.path.join(self.data_path, subject, session, "anat")
        self.logger = logger
        self.running = ""
        self.logger.info(f"STARTING PROCESSING OF SUBJECT {subject}, SESSION {session}")

        file_dict = {
            "inv2": "*inv-2_MP2RAGE*.nii.gz",
            "uni": "*uni_MP2RAGE*.nii.gz",
            "t1map": "*t1map_MP2RAGE*.nii.gz",
            "t2map": "*t2map.nii*.gz",
            "mprage": "*t1w*.nii.gz",
            "flair": "*flair*.nii.gz",
        }

        self.files = defaultdict(None)
        for key, value in file_dict.items():
            self.files[key] = self.find_files(os.path.join(self.session_path, value))

    @staticmethod
    def find_files(glob_string: str):
        output = glob.glob(glob_string)
        return output[0] if len(output) == 1 else None

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

        # Always call voxelwise_zscores — it raises FileExistsError when already done,
        # which populates .output needed for the paths below.
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

        # 2026-05-26-UPDATE
        if self.files["t2map"] is None:
            os.rename(
                os.path.join(output_path, "T1ZscoreMap.nii.gz"),
                os.path.join(output_path, "T1ZscoreMap_SC.nii.gz"),
            )
            os.rename(
                os.path.join(output_path, "T1prior.nii.gz"),
                os.path.join(output_path, "T1_prior_SC.nii.gz"),
            )
        outputs = {
            "t1_zscores_sst": os.path.join(
                voxelwise_zscores.output.get("transformix_T1_template", ""),
                "result.nii.gz",
            ),
            "t2_zscores_sst": os.path.join(
                voxelwise_zscores.output.get("transformix_T2_template", ""),
                "result.nii.gz",
            ),
            "t2overt1_zscores_sst": os.path.join(
                voxelwise_zscores.output.get("transformix_T2OverT1_template", ""),
                "T2OverT1map_sst.nii.gz",
            ),
            "t1_zscores": os.path.join(output_path, "T1ZscoreMap_SC.nii.gz"),
            "t2_zscores": os.path.join(output_path, "T2ZscoreMap_SC.nii.gz"),
            "t2overt1_zscores": os.path.join(
                output_path, "T2OverT1ZscoreMap_SC.nii.gz"
            ),
            "prior_in_mp2rage": os.path.join(output_path, "T1_prior_SC.nii.gz"),
            "tp_uni_to_sst": os.path.join(
                voxelwise_zscores.output.get("elastix_uni_template", ""),
                "TransformParameters.1.txt",
            ),
            "t1_zscores_wholebrain": os.path.join(
                voxelwise_zscores.output.get("zscores", ""),
                "T1_zscores_wholebrain_sst.nii.gz",
            ),
            "t2_zscores_wholebrain": os.path.join(
                voxelwise_zscores.output.get("zscores", ""),
                "T2_zscores_wholebrain_sst.nii.gz",
            ),
            "t2overt1_zscores_wholebrain": os.path.join(
                voxelwise_zscores.output.get("zscores", ""),
                "T2OverT1_zscores_wholebrain_sst.nii.gz",
            ),
            "reoriented_inv2": os.path.join(
                voxelwise_zscores.output.get("reorient", ""), "inv2_reoriented.nii.gz"
            ),
        }

        for key, value in outputs.items():
            self.files[key] = value if os.path.isfile(value) else None

    @log_execution_time
    def run_morphobox(self):
        # --Morphobox segmentation
        output_path = self.output_path.format(f"morphobox_v{VERSION_DL_MORPHOBOX}")
        os.makedirs(output_path, exist_ok=True)

        # output_file = os.path.join(output_path, f'label_v{VERSION_DL_MORPHOBOX}_{self.subject}_{self.session}_T1w.nii.gz')
        output_files = [f for f in os.listdir(output_path) if f.startswith("dl_label")]
        output_file = (
            os.path.join(output_path, output_files[0]) if len(output_files) == 1 else ""
        )

        if not os.path.isfile(output_file):
            # ------------------------------ NEW DL-MorphoBox ------------------------------
            # invoke the executable with subprocess
            # create list that contains the command we would run from the command line
            cmd_list = [
                PATH_DL_MORPHO_EXE,
                "--input_path",
                f"{self.files['uni']}",
                "--out_dir",
                output_path,
                "--volume_type",
                "mp2rage",
                "--magnetic_field_strength",
                "3",
                "--mac_address",
                "04:7C:16:5D:B0:19",
                "--patient_age",
                f"{self.patient_age}",
                "--patient_sex",
                f"{self.patient_sex}",
                "--compute_zscores",
                "False",
                "--save_skull_stripping_outputs",
                "True",
                "--save_outputs_to_disk",
                "True",
                "--save_zscore_maps",
                "False",
                "--save_dev_map",
                "False",
                "--save_regional_T1_dev_map",
                "False",
                "--force_cpu",
                "True",
            ]

            # run cmd_list with subprocess.run
            result = subprocess.run(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            # Check if the command was successful (return code 0)
            if result.returncode == 0:
                # Redirect the output to a text file
                out_path_log = os.path.join(output_path, "morphobox_log.txt")
                with open(out_path_log, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
            else:
                self.logger.error(f"\nERROR: {result.stderr}")

        output_files = [f for f in os.listdir(output_path) if f.startswith("dl_label")]
        assert (
            len(output_files) == 1
        ), f"Expected exactly one file starting with 'dl_label' in {output_path}, found {len(output_files)}"

        self.files["MB_label_map"] = output_file

    @log_execution_time
    def run_concord(self):
        version = metadata.version("shs_acit_concord")
        output_path = self.output_path.format(f"concord_v{version}")
        os.makedirs(output_path, exist_ok=True)
        concord_process([self.files["inv2"], self.files["uni"]], output_path)

    @log_execution_time
    def run_registration_flair_to_mp2rage(self):
        output_path = self.output_path.format("registration_flair_to_MP2RAGE_v1.0")
        transform_parameter_file = os.path.join(
            output_path, "TransformParameters.0.txt"
        )
        if not os.path.isfile(transform_parameter_file):
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
            transform_parameter_file
            if os.path.isfile(transform_parameter_file)
            else None
        )

    @log_execution_time
    def run_registration_lesion_map_to_mp2rage(self):
        output_path = self.output_path.format("registration_lesion_to_MP2RAGE_v1.0")
        os.makedirs(output_path, exist_ok=True)
        run_transformix(
            moving=self.files["lesion_map"],
            transform_param_file=self.files[
                "transform_parameter_file_flair_to_mp2rage"
            ],
            output_path=output_path,
            output_filename="lesion_map_reg_to_mp2rage.nii.gz",
            param_to_modify={
                "FinalBSplineInterpolationOrder": (None, "0")
            },  # nearest neighbour interpolation (lesion mask is binary)
            save=True,
        )

        output_file = os.path.join(output_path, "lesion_map_reg_to_mp2rage.nii.gz")
        self.files["lesion_map_in_MP2RAGE"] = (
            output_file if os.path.isfile(output_file) else None
        )

    @log_execution_time
    def run_registration_lesion_map_to_sst(self):
        output_path = self.output_path.format("registration_lesion_map_to_SST")
        output_file = os.path.join(output_path, "lesion_map_reg_to_sst.nii.gz")
        if not os.path.isfile(output_file):
            os.makedirs(output_path, exist_ok=True)
            run_transformix(
                moving=self.files["lesion_map_in_MP2RAGE"],
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

        if not os.path.isfile(output_file):
            os.makedirs(output_path, exist_ok=True)

            flair = sitk.ReadImage(self.files["flair"])
            results = phoenix_wmh.process_case(current=flair, return_cross=True)
            sitk.WriteImage(results["wmh_cross"], output_file)

        self.files["lesion_map"] = output_file if os.path.isfile(output_file) else None

    @log_execution_time
    def run_perilesional_rings(self):
        # output_path = self.output_path.format('perilesional_rings_v1.0')
        version = metadata.version("shs_acit_quant_periles")
        output_path = self.output_path.format(f"quant_periles_t1_v{version}")
        os.makedirs(output_path, exist_ok=True)
        ra = RingAnalyzer(
            output_path=output_path,
            lesions_path=self.files["lesion_map_in_MP2RAGE"],
            qmap_path=self.files["t1_zscores"],
            wm_sc_mask_path=self.files["prior_in_mp2rage"],
        )
        dist_stat_t1 = ra.compute_distance_statistics(label="my label")

        dist_stat_t2 = None
        if self.files["t2_zscores"] is not None:
            output_path_t2 = self.output_path.format(f"quant_periles_t2_v{version}")
            os.makedirs(output_path_t2, exist_ok=True)
            ra = RingAnalyzer(
                output_path=output_path_t2,
                lesions_path=self.files["lesion_map_in_MP2RAGE"],
                qmap_path=self.files["t2_zscores"],
                wm_sc_mask_path=self.files["prior_in_mp2rage"],
            )
            dist_stat_t2 = ra.compute_distance_statistics(label="my label")

        dist_stat_t2overt1 = None
        if self.files["t2overt1_zscores"] is not None:
            output_path_t2overt1 = self.output_path.format(
                f"quant_periles_t2overt1_v{version}"
            )
            os.makedirs(output_path_t2overt1, exist_ok=True)
            ra = RingAnalyzer(
                output_path=output_path_t2overt1,
                lesions_path=self.files["lesion_map_in_MP2RAGE"],
                qmap_path=self.files["t2overt1_zscores"],
                wm_sc_mask_path=self.files["prior_in_mp2rage"],
            )
            dist_stat_t2overt1 = ra.compute_distance_statistics(label="my label")

        return dist_stat_t1, dist_stat_t2, dist_stat_t2overt1

    @log_execution_time
    def run_periventricular(self, perivent_zscores: Perivent_zscores):
        # output_path_t1 = self.output_path.format("periventricular_t1_v1.0")
        pkg_version = metadata.version("shs_acit_quant_perivent")
        output_path_t1 = self.output_path.format(f"quant_perivent_t1_v{pkg_version}")
        os.makedirs(output_path_t1, exist_ok=True)

        perivent_zscores.compute_qvalues_in_perivent(
            self.files["t1_zscores_sst"],
            self.files["lesion_map_in_sst"],
            output_path=output_path_t1,
        )

        if self.files["t2_zscores_sst"] is not None:
            output_path_t2 = self.output_path.format(
                f"quant_perivent_t2_v{pkg_version}"
            )
            os.makedirs(output_path_t2, exist_ok=True)

            perivent_zscores.compute_qvalues_in_perivent(
                self.files["t2_zscores_sst"],
                self.files["lesion_map_in_sst"],
                output_path=output_path_t2,
            )

        if self.files["t2overt1_zscores_sst"] is not None:
            output_path_t2overt1 = self.output_path.format(
                f"quant_perivent_t2overt1_v{pkg_version}"
            )
            os.makedirs(output_path_t2overt1, exist_ok=True)

            perivent_zscores.compute_qvalues_in_perivent(
                self.files["t2overt1_zscores_sst"],
                self.files["lesion_map_in_sst"],
                output_path=output_path_t2overt1,
            )

    @log_execution_time
    def run_quant_regions(self, zscores_stats_in_mb_regions: ZscoresMBRegions):
        pkg_version = metadata.version("shs_acit_quant_regions")
        if self.files["t1_zscores"] is not None:
            output_path = self.output_path.format(f"quant_regions_t1_v{pkg_version}")
            os.makedirs(output_path, exist_ok=True)

            zscores_stats_in_mb_regions.compute_qvalues_in_regions(
                qmap_path=self.files["t1_zscores"],
                label_path=self.files["MB_label_map"],
                lesion_mask_path=self.files["lesion_map_in_MP2RAGE"],
                wm_sc_mask_path=self.files["prior_in_mp2rage"],
                output_path=output_path,
            )

        if self.files["t2_zscores"] is not None:
            output_path_t2 = self.output_path.format(f"quant_regions_t2_v{pkg_version}")
            os.makedirs(output_path_t2, exist_ok=True)

            zscores_stats_in_mb_regions.compute_qvalues_in_regions(
                qmap_path=self.files["t2_zscores"],
                label_path=self.files["MB_label_map"],
                lesion_mask_path=self.files["lesion_map_in_MP2RAGE"],
                wm_sc_mask_path=self.files["prior_in_mp2rage"],
                output_path=output_path_t2,
            )

        if self.files["t2overt1_zscores"] is not None:
            output_path_t2overt1 = self.output_path.format(
                f"quant_regions_t2overt1_v{pkg_version}"
            )
            os.makedirs(output_path_t2overt1, exist_ok=True)

            zscores_stats_in_mb_regions.compute_qvalues_in_regions(
                qmap_path=self.files["t2overt1_zscores"],
                label_path=self.files["MB_label_map"],
                lesion_mask_path=self.files["lesion_map_in_MP2RAGE"],
                wm_sc_mask_path=self.files["prior_in_mp2rage"],
                output_path=output_path_t2overt1,
            )

    @log_execution_time
    def run_tracts(self, wm_lesions_zscores: WMLesions_zscores):
        pkg_version = metadata.version("shs_acit_quant_wmtracts")
        if self.files["t1_zscores_wholebrain"] is not None:
            output_path = self.output_path.format(f"quant_wmtracts_t1_v{pkg_version}")
            os.makedirs(output_path, exist_ok=True)
            wm_lesions_zscores.compute_qvalues_in_tracts(
                self.files["t1_zscores_wholebrain"],
                self.files["lesion_map_in_sst"],
                output_path=output_path,
            )

        if self.files["t2_zscores_wholebrain"] is not None:
            output_path_t2 = self.output_path.format(
                f"quant_wmtracts_t2_v{pkg_version}"
            )
            os.makedirs(output_path_t2, exist_ok=True)

            wm_lesions_zscores.compute_qvalues_in_tracts(
                self.files["t2_zscores_wholebrain"],
                self.files["lesion_map_in_sst"],
                output_path=output_path_t2,
            )

        if self.files["t2overt1_zscores_wholebrain"] is not None:
            output_path_t2overt1 = self.output_path.format(
                f"quant_wmtracts_t2overt1_v{pkg_version}"
            )
            os.makedirs(output_path_t2overt1, exist_ok=True)

            wm_lesions_zscores.compute_qvalues_in_tracts(
                self.files["t2overt1_zscores_wholebrain"],
                self.files["lesion_map_in_sst"],
                output_path=output_path_t2overt1,
            )


class PatientNameSex:
    """Class used to extract patient age and sex from a demographics table. The class associates
    the subject and session to the corresponding age and sex.

    Parameters
    ----------
    table_path : str
        Path to the Excel file containing the demographics table.
    """

    def __init__(self, table_path: str):
        """Initialisation

        Parameters
        ----------
        table_path : str
            Path to the Excel file containing the demographics table.
            Should have the following columns: PatientID, AcquisitionDate, PatientSex, PatientAge.
        """

        table = pd.read_excel(table_path)

        # pick unique combination of sub and ses
        self.preprocessing_table = table[
            ["PatientID", "AcquisitionDate", "PatientSex", "PatientAge"]
        ].drop_duplicates()

    def __call__(self, subject: str, session: str):
        patient_sex = self.preprocessing_table["PatientSex"][
            (self.preprocessing_table["PatientID"] == int(subject[4:]))
            & (self.preprocessing_table["AcquisitionDate"] == int(session[4:]))
        ].values[0]
        patient_age = int(
            self.preprocessing_table["PatientAge"][
                (self.preprocessing_table["PatientID"] == int(subject[4:]))
                & (self.preprocessing_table["AcquisitionDate"] == int(session[4:]))
            ].values[0]
        )
        return patient_age, patient_sex


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
    # today = datetime.today().strftime("%Y_%m_%d")
    today = '2026_07_02'
    # define paths
    bids_dir = r"E:\Project_DCE\DCE_PRG_NEW\bids_test_hc"  # TODO: change to actual path
    bids_dir = r"E:\Project_DCE\DCE_PRG_NEW\bids"
    # for future put preprocess table to folder derivatives and then change following path
    # create derivatives path and folder
    derivatives_path = os.path.join(bids_dir, "derivatives", today)
    os.makedirs(derivatives_path, exist_ok=True)
    demographics = PatientNameSex(
        glob.glob(os.path.join(bids_dir, "bids_demographics_sequences.xlsx"))[0]
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

    # 2026-05-26-UPDATE
    voxelwise_t1_zscores = ZScore(atlas_t1=atlas_t1, logger=logger, writetmp=True)
    voxelwise_t2overt1_zscores = T2OverT1ZScore(
        atlas_t2overt1=atlas_t2overt1,
        atlas_t1=atlas_t1,
        atlas_t2=atlas_t2,
        logger=logger,
        writetmp=True,
    )
    phoenix_wmh = Phoenix()
    wm_lesions_zscores = WMLesions_zscores(prague_atlas, threshold_value=2.0)
    perivent_zscores = Perivent_zscores(prague_atlas, threshold_value=2.0)
    zscores_stats_in_mb_regions = ZscoresMBRegions(logger=logger, threshold_value=2.0)

    subjects = [
        item for item in os.listdir(bids_dir)
        if os.path.isdir(os.path.join(bids_dir, item)) and item.startswith("sub-")
    ]
    subjects.sort()
    # subjects = [ 'sub-144820', 'sub-145098', 'sub-145413', 'sub-145533', 'sub-146016', 'sub-146161']
    for subject in tqdm(subjects):
        sessions = [
            item
            for item in os.listdir(os.path.join(bids_dir, subject))
            if os.path.isdir(os.path.join(bids_dir, subject, item))
            and item.startswith("ses-")
        ]
        for session in sessions:
            patient_age, patient_sex = demographics(subject, session)
            proc = Processing(
                subject=subject,
                session=session,
                data_path=bids_dir,
                derivatives_path=derivatives_path,
                sex=patient_sex,
                age=patient_age,
                logger=logger,
            )

            try:
                # 2026-05-26-UPDATE
                if proc.files["t2map"] is None:
                    proc.run_zscores(voxelwise_zscores=voxelwise_t1_zscores)
                else:
                    proc.run_zscores(voxelwise_zscores=voxelwise_t2overt1_zscores)
                proc.run_morphobox()
                proc.run_wmh(phoenix_wmh=phoenix_wmh)
                proc.run_registration_flair_to_mp2rage()
                proc.run_registration_lesion_map_to_mp2rage()
                proc.run_registration_lesion_map_to_sst()
                # proc.run_concord()
                # proc.run_perilesional_rings()
                # proc.run_tracts(wm_lesions_zscores=wm_lesions_zscores)
                # proc.run_periventricular(perivent_zscores=perivent_zscores)
                proc.run_quant_regions(
                    zscores_stats_in_mb_regions=zscores_stats_in_mb_regions
                )

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
