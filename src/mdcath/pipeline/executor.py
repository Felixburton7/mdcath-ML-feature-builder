# -*- coding: utf-8 -*-
"""
Orchestrates the mdCATH processing pipeline steps.
Includes parallel processing worker function to avoid pickling errors.
"""
import os
import glob
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import importlib
import inspect
import traceback # Import traceback module

# Import package components
from ..config import load_config
from ..io.hdf5_reader import HDF5Reader, HDF5ReaderError
from ..structure.pdb_processor import PDBProcessor
from ..structure.properties import PropertiesCalculator
from ..structure.frame_extractor import extract_and_save_frames
from ..metrics.rmsf import process_domain_rmsf, aggregate_and_average_rmsf, save_rmsf_results
from ..features.builder import FeatureBuilder, FeatureBuilderError
from ..voxel.aposteriori_wrapper import run_aposteriori, VoxelizationError
from ..visualize import plots as viz_plots
from ..utils.parallel import parallel_map
from ..io.writers import save_dataframe_csv

# Import custom exceptions
from ..exceptions import (
    RmsfProcessingError,
    PDBProcessorError,
    PropertiesCalculatorError,
    HDF5ReaderError,
    PipelineExecutionError,
    FeatureBuilderError,
    VoxelizationError,
    ConfigurationError
)

# Define PipelineExecutionError if not defined in exceptions.py
if 'PipelineExecutionError' not in locals():
    class PipelineExecutionError(Exception):
        """Custom exception for pipeline execution errors."""
        pass

# --- Top-level Worker Function for Parallel Processing ---
def _parallel_domain_worker(args_tuple: Tuple[str, Dict, str]) -> Tuple[str, str, Optional[Any], Optional[pd.DataFrame], Optional[str]]:
    """
    Worker function for parallel domain processing. Initializes components internally.

    Args:
        args_tuple (Tuple): Contains (domain_id, config_dict, output_dir)

    Returns:
        Tuple: (domain_id, status, rmsf_result, properties_result, cleaned_pdb_path)
    """
    domain_id, config_dict, output_dir = args_tuple
    status = f"Starting (worker) {domain_id}"
    rmsf_result = None
    properties_result = None
    cleaned_pdb_path = None
    # Configure logging within worker if needed (basic for now)
    # logging.basicConfig(level=logging.INFO) # Or pass level from main config

    try:
        # Initialize components INSIDE the worker to avoid pickling issues
        pdb_processor = PDBProcessor(config_dict.get('processing', {}).get('pdb_cleaning', {}))
        properties_calculator = PropertiesCalculator(config_dict.get('processing', {}).get('properties_calculation', {}))

        input_cfg = config_dict.get('input', {})
        mdcath_folder = input_cfg.get('mdcath_folder')
        h5_path = os.path.join(mdcath_folder, f"mdcath_dataset_{domain_id}.h5")
        reader = HDF5Reader(h5_path)
        status = f"Read HDF5 OK"

        pdb_string = reader.get_pdb_string()
        if not pdb_string: status = "Failed PDB Read"; raise ValueError("Could not read PDB string")

        pdb_out_dir = os.path.join(output_dir, "pdbs")
        os.makedirs(pdb_out_dir, exist_ok=True)
        cleaned_pdb_path = os.path.join(pdb_out_dir, f"{domain_id}.pdb")

        if not pdb_processor.clean_pdb_string(pdb_string, cleaned_pdb_path):
             status = "Failed PDB Clean"; raise PDBProcessorError("PDB cleaning failed")
        status = "PDB Clean OK"

        properties_result = properties_calculator.calculate_properties(cleaned_pdb_path)
        if properties_result is None or properties_result.empty:
             status = "Failed Properties Calc"; raise PropertiesCalculatorError("Property calculation failed")
        status = "Properties Calc OK"

        rmsf_result_tuple = process_domain_rmsf(domain_id, reader, config_dict)
        if rmsf_result_tuple is None:
            status = "Failed RMSF Proc"; raise RmsfProcessingError("RMSF processing failed")

        rmsf_result = (rmsf_result_tuple[0].copy(), rmsf_result_tuple[1])
        status = "RMSF Proc OK"

        status = "Success"

    except (FileNotFoundError, HDF5ReaderError, ValueError) as e:
         status = status if status.startswith("Failed") else "Failed HDF5 Read/Access"
         logging.error(f"Worker error {domain_id} (Read): {e}")
    except PDBProcessorError as e:
         status = status if status.startswith("Failed") else "Failed PDB Clean"
         logging.error(f"Worker error {domain_id} (Clean): {e}")
    except PropertiesCalculatorError as e:
         status = status if status.startswith("Failed") else "Failed Properties Calc"
         logging.error(f"Worker error {domain_id} (Properties): {e}")
    except RmsfProcessingError as e:
         status = status if status.startswith("Failed") else "Failed RMSF Proc"
         logging.error(f"Worker error {domain_id} (RMSF): {e}")
    except Exception as e:
        status = f"Failed Unexpected (Worker): {type(e).__name__}"
        logging.error(f"Unexpected worker error {domain_id}: {e}")
        # Log traceback if needed (might be complex from worker)
        # logging.debug(traceback.format_exc()) # Might not work well here

    return domain_id, status, rmsf_result, properties_result, cleaned_pdb_path


# --- PipelineExecutor Class ---
class PipelineExecutor:
    """ Manages and executes the mdCATH processing pipeline. """
    def __init__(self, config_dict: Dict[str, Any]):
        """ Initializes the executor with a configuration dictionary. """
        try:
            self.config = config_dict
            self.output_dir = self.config.get('output', {}).get('base_dir', './outputs')
            self.num_cores = self.config.get('performance', {}).get('num_cores', 0)
            cpu_count = os.cpu_count()
            if self.num_cores <= 0:
                 self.num_cores = max(1, cpu_count - 2 if cpu_count is not None and cpu_count > 2 else 1)
            else:
                 self.num_cores = self.num_cores
            logging.info(f"Determined number of cores to use: {self.num_cores}")

            # Initialize components that are safe to initialize here (stateless or used sequentially)
            # self.pdb_processor = PDBProcessor(...) # Initializing in worker now
            # self.properties_calculator = PropertiesCalculator(...) # Initializing in worker now

            # Pipeline state
            self.domain_list: List[str] = []
            self.domain_status: Dict[str, str] = {}
            self.all_domain_rmsf_results: Dict[str, Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]] = {}
            self.all_domain_properties: Dict[str, Optional[pd.DataFrame]] = {}
            self.cleaned_pdb_paths: Dict[str, Optional[str]] = {}
            self.voxel_output_file: Optional[str] = None
            self.agg_replica_data: Optional[Dict] = None
            self.agg_replica_avg_data: Optional[Dict] = None
            self.agg_overall_avg_data: Optional[pd.DataFrame] = None
            self.all_feature_dfs: Dict[str, pd.DataFrame] = {}

        except Exception as e:
            logging.exception(f"Failed during PipelineExecutor initialization: {e}")
            raise PipelineExecutionError(f"Initialization failed: {e}") from e

    def _get_domain_list(self) -> List[str]:
        """Determines the list of domain IDs to process."""
        # (Keep this method as is)
        input_cfg = self.config.get('input', {})
        configured_domains = input_cfg.get('domain_ids')
        mdcath_folder = input_cfg.get('mdcath_folder')

        if isinstance(configured_domains, list) and configured_domains:
            logging.info(f"Using specified list of {len(configured_domains)} domains.")
            return configured_domains
        elif mdcath_folder and os.path.isdir(mdcath_folder):
            pattern = os.path.join(mdcath_folder, "mdcath_dataset_*.h5")
            logging.debug(f"Searching for HDF5 files with pattern: {pattern}")
            h5_files = glob.glob(pattern)
            if not h5_files:
                 raise PipelineExecutionError(f"No 'mdcath_dataset_*.h5' files found in specified directory: {mdcath_folder}")

            domains = []
            for h5_file in h5_files:
                basename = os.path.basename(h5_file)
                if basename.startswith("mdcath_dataset_") and basename.endswith(".h5"):
                    domain_id = basename[len("mdcath_dataset_"):-len(".h5")]
                    if domain_id: domains.append(domain_id)
                    else: logging.warning(f"Skipping file with potentially empty domain ID: {h5_file}")
                else: logging.warning(f"Skipping file not matching expected pattern: {h5_file}")

            if not domains: raise PipelineExecutionError(f"No valid domains extracted from filenames in: {mdcath_folder}")

            logging.info(f"Found {len(domains)} domains in {mdcath_folder}.")
            return sorted(domains)
        else:
            raise PipelineExecutionError("Config must provide 'input.domain_ids' list or valid 'input.mdcath_folder'.")

    # _process_single_domain is no longer called directly by parallel_map
    # Keep it for potential sequential use or reference if needed
    def _process_single_domain(self, domain_id: str) -> Tuple[str, str, Optional[Any], Optional[pd.DataFrame], Optional[str]]:
         """ Sequential processing for a single domain (used as fallback). """
         # This essentially duplicates the worker function logic
         # Can call the worker function even in sequential mode for consistency
         return _parallel_domain_worker((domain_id, self.config, self.output_dir))

    def _extract_all_frames(self):
        """Extracts frames sequentially after initial processing."""
        # (Keep this method as is)
        frame_cfg = self.config.get('processing', {}).get('frame_selection', {})
        if frame_cfg.get('num_frames', 0) <= 0:
            logging.info("Frame extraction skipped (num_frames <= 0).")
            return

        logging.info("Starting frame extraction...")
        domains_to_process = [did for did, status in self.domain_status.items() if status == "Success"]
        if not domains_to_process:
             logging.warning("No successfully processed domains available for frame extraction.")
             return

        input_cfg = self.config.get('input', {})
        mdcath_folder = input_cfg.get('mdcath_folder')
        num_success = 0
        num_failed = 0

        for domain_id in tqdm(domains_to_process, desc="Extracting Frames", disable=not self.config.get('logging',{}).get('show_progress_bars', True)):
            domain_frames_saved_flag = False
            try:
                 h5_path = os.path.join(mdcath_folder, f"mdcath_dataset_{domain_id}.h5")
                 if not os.path.exists(h5_path):
                      logging.error(f"HDF5 file not found for frame extraction: {h5_path}")
                      num_failed += 1; continue

                 reader = HDF5Reader(h5_path)
                 cleaned_pdb_path = self.cleaned_pdb_paths.get(domain_id)
                 if not cleaned_pdb_path or not os.path.exists(cleaned_pdb_path):
                      logging.warning(f"Skipping frame extraction for {domain_id}: Missing cleaned PDB '{cleaned_pdb_path}'.")
                      num_failed+=1; continue

                 temps = reader.get_available_temperatures()
                 for temp in temps:
                     replicas = reader.get_available_replicas(temp)
                     for rep in replicas:
                          coords_all = reader.get_coordinates(temp, rep, frame_index=-999)
                          if coords_all is None:
                               logging.warning(f"No coordinates found for {domain_id}, T={temp}, R={rep}. Skipping."); continue

                          rmsd_data = reader.get_scalar_traj(temp, rep, 'rmsd')
                          gyration_data = reader.get_scalar_traj(temp, rep, 'gyrationRadius')

                          success = extract_and_save_frames(
                               domain_id, coords_all, cleaned_pdb_path, self.output_dir,
                               self.config, rmsd_data, gyration_data, str(temp), str(rep)
                          )
                          if success: domain_frames_saved_flag = True

                 if domain_frames_saved_flag: num_success += 1
                 else: num_failed += 1

            except Exception as e:
                 logging.error(f"Error during frame extraction loop for {domain_id}: {e}", exc_info=True)
                 num_failed += 1

        logging.info(f"Frame extraction complete. Domains with frames saved: {num_success}, Domains failed: {num_failed}")

    def run(self):
        """Executes the full processing pipeline."""
        try:
            logging.info("Starting mdCATH processing pipeline...")
            logging.info(f"Using output directory: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)

            # --- Step 1: Get Domain List ---
            self.domain_list = self._get_domain_list()
            if not self.domain_list: logging.warning("No domains found/specified."); return

            # --- Step 2: Initial Domain Processing ---
            logging.info(f"Processing {len(self.domain_list)} domains using up to {self.num_cores} cores...")
            results = []
            show_progress = self.config.get('logging',{}).get('show_progress_bars', True)
            if self.num_cores > 1 and len(self.domain_list) > 1:
                logging.info(f"Using parallel processing with {self.num_cores} cores.")
                try:
                     # Prepare arguments for the worker function
                     worker_args = [(domain_id, self.config, self.output_dir) for domain_id in self.domain_list]
                     results = parallel_map(_parallel_domain_worker, worker_args, # Call the top-level worker
                                            num_cores=self.num_cores, use_progress_bar=show_progress,
                                            desc="Processing Domains")
                except Exception as parallel_e:
                     logging.error(f"Parallel processing failed: {parallel_e}. Falling back to sequential.")
                     # *** ROBUST LOGGING FIX ***
                     if logging.getLogger().isEnabledFor(logging.DEBUG):
                         try: logging.debug("Traceback for parallel processing failure:", exc_info=True)
                         except TypeError: logging.debug(f"Traceback:\n{traceback.format_exc()}") # Use traceback module
                     # Use the sequential instance method as fallback
                     results = [self._process_single_domain(did) for did in tqdm(self.domain_list, desc="Processing (Sequential Fallback)", disable=not show_progress)]
            else:
                logging.info("Using sequential processing.")
                results = [self._process_single_domain(did) for did in tqdm(self.domain_list, desc="Processing Domains", disable=not show_progress)]

            # Collect results (remains the same)
            for result in results:
                 if result and isinstance(result, tuple) and len(result) == 5:
                    domain_id, status, rmsf_res, prop_res, pdb_path = result
                    self.domain_status[domain_id] = status
                    if status == "Success":
                        self.all_domain_rmsf_results[domain_id] = rmsf_res; self.all_domain_properties[domain_id] = prop_res; self.cleaned_pdb_paths[domain_id] = pdb_path
                 else: logging.error(f"Invalid result from domain processing: {result}")

            successful_domains = [d for d, s in self.domain_status.items() if s == "Success"]
            logging.info(f"Initial processing complete. Successful domains: {len(successful_domains)}/{len(self.domain_list)}")
            if not successful_domains:
                 logging.error("No domains processed successfully. Cannot proceed further."); self._generate_status_visualization(); return

            # --- Step 3: Aggregate RMSF ---
            # (Keep as is)
            logging.info("Aggregating RMSF data...")
            try:
                self.agg_replica_data, self.agg_replica_avg_data, self.agg_overall_avg_data = aggregate_and_average_rmsf(self.all_domain_rmsf_results, self.config)
            except Exception as e: logging.exception(f"Error during RMSF aggregation: {e}"); self.agg_replica_data, self.agg_replica_avg_data, self.agg_overall_avg_data = None, None, None


            # --- Step 4: Save RMSF Results ---
            # (Keep as is)
            try: save_rmsf_results(self.output_dir, self.config, self.agg_replica_data, self.agg_replica_avg_data, self.agg_overall_avg_data)
            except Exception as e: logging.exception(f"Error saving RMSF results: {e}")


            # --- Step 5: Build ML Features ---
            # (Keep as is, including robust logging)
            logging.info("Building ML feature sets...")
            try:
                 valid_replica_avg = {k: v for k, v in (self.agg_replica_avg_data or {}).items() if isinstance(v, pd.DataFrame) and not v.empty}
                 valid_overall_avg = self.agg_overall_avg_data if isinstance(self.agg_overall_avg_data, pd.DataFrame) and not self.agg_overall_avg_data.empty else None
                 valid_properties = {k: v for k, v in self.all_domain_properties.items() if isinstance(v, pd.DataFrame) and not v.empty}

                 if (not valid_replica_avg and valid_overall_avg is None) or not valid_properties:
                     if not valid_replica_avg and valid_overall_avg is None: logging.warning("Skipping feature building: No valid RMSF average data.")
                     if not valid_properties: logging.warning("Skipping feature building: No valid structure properties.")
                     self.all_feature_dfs = {}
                 else:
                     feature_builder = FeatureBuilder(self.config.get('processing', {}).get('feature_building', {}), valid_replica_avg, valid_overall_avg, valid_properties)
                     self.all_feature_dfs = {}
                     ml_features_dir = os.path.join(self.output_dir, "ML_features")
                     temps = list(valid_replica_avg.keys())
                     for temp_str in tqdm(temps, desc="Building Temp Features", disable=not show_progress):
                          features_df = feature_builder.build_features(temperature=temp_str)
                          if features_df is not None:
                               self.all_feature_dfs[temp_str] = features_df; save_dataframe_csv(features_df, os.path.join(ml_features_dir, f"final_dataset_temperature_{temp_str}.csv"))
                     overall_features_df = feature_builder.build_features(temperature=None)
                     if overall_features_df is not None:
                          self.all_feature_dfs["average"] = overall_features_df; save_dataframe_csv(overall_features_df, os.path.join(ml_features_dir, "final_dataset_temperature_average.csv"))
            except Exception as e:
                 logging.error(f"Error during feature building: {e}")
                 if logging.getLogger().isEnabledFor(logging.DEBUG):
                     try: logging.debug("Traceback:", exc_info=True)
                     except TypeError: logging.debug(f"Traceback:\n{traceback.format_exc()}")
                 self.all_feature_dfs = {}


            # --- Step 6: Extract Frames ---
            # (Keep as is)
            try: self._extract_all_frames()
            except Exception as e: logging.exception(f"Error during frame extraction: {e}")


            # --- Step 7: Voxelization ---
            # (Keep as is, including robust logging)
            logging.info("Running voxelization...")
            try:
                 cleaned_pdb_dir = os.path.join(self.output_dir, "pdbs")
                 voxel_config = self.config.get('processing', {}).get('voxelization', {})
                 if not os.path.isdir(cleaned_pdb_dir) or not os.listdir(cleaned_pdb_dir): logging.warning("Skipping Voxelization: Cleaned PDB dir empty/missing.")
                 elif not voxel_config.get("enabled", True): logging.info("Voxelization skipped by config.")
                 else:
                     voxel_success = run_aposteriori(voxel_config, self.output_dir, cleaned_pdb_dir)
                     if voxel_success:
                         voxel_output_name = voxel_config.get("output_name", "mdcath_voxelized")
                         potential_path = os.path.join(self.output_dir, "voxelized", f"{voxel_output_name}.hdf5")
                         if os.path.exists(potential_path): self.voxel_output_file = potential_path
                         else: logging.warning(f"Voxelization command finished, but output file not found: {potential_path}")
            except Exception as e:
                 logging.error(f"Unexpected error during voxelization: {e}")
                 if logging.getLogger().isEnabledFor(logging.DEBUG):
                     try: logging.debug("Traceback:", exc_info=True)
                     except TypeError: logging.debug(f"Traceback:\n{traceback.format_exc()}")


            # --- Step 8: Generate Visualizations ---
            # (Keep as is)
            if self.config.get('visualization', {}).get('enabled', True):
                 try: self._generate_visualizations() # Call refactored method
                 except Exception as e:
                     logging.error(f"Error during visualization generation: {e}")
                     if logging.getLogger().isEnabledFor(logging.DEBUG):
                         try: logging.debug("Traceback:", exc_info=True)
                         except TypeError: logging.debug(f"Traceback:\n{traceback.format_exc()}")
            else: logging.info("Visualization generation skipped by configuration.")

            logging.info("Pipeline finished.")

        except Exception as e:
             logging.error(f"An unexpected error occurred during pipeline execution: {e}")
             # Apply robust logging fix here as well
             if logging.getLogger().isEnabledFor(logging.DEBUG):
                 try: logging.debug("Traceback for pipeline execution failure:", exc_info=True)
                 except TypeError: logging.debug(f"Traceback:\n{traceback.format_exc()}")
             raise PipelineExecutionError(f"Pipeline failed unexpectedly: {e}") from e

    # --- run_plot Helper ---
    def run_plot(self, func, *args, **kwargs):
        """ Helper function to run a plotting function with error handling and data validation. """
        arg_name = func.__name__
        logging.debug(f"Attempting to generate plot: {arg_name}")

        # Basic data validation
        all_args_valid = True
        for i, data_arg in enumerate(args):
            arg_desc = f"positional argument {i+1}"
            if data_arg is None: logging.warning(f"Skipping plot {arg_name}: Required {arg_desc} is None."); return
            if isinstance(data_arg, pd.DataFrame) and data_arg.empty: logging.warning(f"Skipping plot {arg_name}: Required DataFrame {arg_desc} is empty."); return
            if isinstance(data_arg, (dict, list, tuple)) and not data_arg: logging.warning(f"Skipping plot {arg_name}: Required {type(data_arg).__name__} {arg_desc} is empty."); return

        if not all_args_valid: return # Exit if validation failed (though loop structure makes this redundant)

        try:
            plot_kwargs = kwargs.copy()
            func_sig = inspect.signature(func)
            func_params = func_sig.parameters

            if 'output_dir' in func_params: plot_kwargs.setdefault('output_dir', self.output_dir)
            if 'viz_config' in func_params: plot_kwargs.setdefault('viz_config', self.config.get('visualization', {}))

            accepted_kwargs = {k: v for k, v in plot_kwargs.items() if k in func_params}
            if any(p.kind == p.VAR_KEYWORD for p in func_params.values()): final_kwargs = plot_kwargs
            else: final_kwargs = accepted_kwargs

            func(*args, **final_kwargs)

        except Exception as e:
            # *** ROBUST LOGGING FIX ***
            error_message = f"Failed to generate plot {func.__name__}: {e}"
            logging.error(error_message)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                try: logging.debug(f"Traceback for plot failure ({func.__name__}):", exc_info=True)
                except TypeError: logging.debug(f"Traceback for plot failure ({func.__name__}):\n{traceback.format_exc()}")

    # --- _generate_status_visualization Helper ---
    def _generate_status_visualization(self):
         """Helper to generate only the status plot, e.g., on early exit."""
         if self.config.get('visualization', {}).get('enabled', True) and self.domain_status:
             logging.info("Generating processing status visualization...")
             try:
                  avg_feature_df_for_status = self.all_feature_dfs.get("average", pd.DataFrame())
                  replica_avg_data_for_status = self.agg_replica_avg_data if self.agg_replica_avg_data is not None else {}
                  # Call the appropriate function - assuming create_summary_plot handles status
                  self.run_plot(viz_plots.create_summary_plot,
                                replica_avg_data_for_status,
                                avg_feature_df_for_status,
                                self.domain_status)
             except Exception as e:
                  logging.error(f"Failed to generate status plot (via summary): {e}")
                  if logging.getLogger().isEnabledFor(logging.DEBUG):
                      try: logging.debug("Traceback:", exc_info=True)
                      except TypeError: logging.debug(f"Traceback:\n{traceback.format_exc()}")

    # --- _generate_visualizations Method ---
    def _generate_visualizations(self):
          """Generates all configured plots by calling functions from viz_plots."""
          logging.info("Generating visualizations...")

          # --- Prepare Data References ---
          replica_avg_data = self.agg_replica_avg_data if self.agg_replica_avg_data is not None else {}
          avg_feature_df = self.all_feature_dfs.get("average", pd.DataFrame())
          domain_status_data = self.domain_status if self.domain_status is not None else {}
          voxel_output = self.voxel_output_file
          combined_replica_data = self.agg_replica_data if self.agg_replica_data else {}
          overall_avg_data = self.agg_overall_avg_data if self.agg_overall_avg_data is not None else pd.DataFrame()
          feature_dfs_dict = self.all_feature_dfs if self.all_feature_dfs else {} # Pass the whole dict if needed
          pdb_results_placeholder = {} # Need to collect PDB paths during main loop
          domain_results_placeholder = self.domain_status # Use status dict as placeholder for domain results

          # --- Define Plotting Tasks ---
          # Use the function names from the user-provided plots.py
          plotting_tasks = [
              (viz_plots.create_temperature_summary_heatmap, [replica_avg_data], {}),
              (viz_plots.create_temperature_average_summary, [avg_feature_df], {}),
              # Pass both replica and overall avg to the function that creates violin + combined histogram
              (viz_plots.create_rmsf_distribution_plots, [replica_avg_data, overall_avg_data], {}),
              (viz_plots.create_amino_acid_rmsf_plot, [avg_feature_df], {}), # Uses avg_df now
              (viz_plots.create_amino_acid_rmsf_plot_colored, [avg_feature_df], {}),
              (viz_plots.create_replica_variance_plot, [combined_replica_data], {}),
              (viz_plots.create_dssp_rmsf_correlation_plot, [avg_feature_df], {}), # Uses avg_df
              (viz_plots.create_feature_correlation_plot, [avg_feature_df], {}), # Uses avg_df
              # (viz_plots.create_frames_visualization, [pdb_results_placeholder, self.config, domain_results_placeholder], {}), # Requires more data storage
              (viz_plots.create_ml_features_plot, [avg_feature_df], {}), # Uses avg_df
              (viz_plots.create_summary_plot, [replica_avg_data, avg_feature_df, domain_status_data], {}), # Pass avg_df
              (viz_plots.create_additional_ml_features_plot, [avg_feature_df], {}), # Uses avg_df
              # Ensure these new functions exist in plots.py
              (getattr(viz_plots, 'create_rmsf_density_plots', None), [avg_feature_df], {}),
              (getattr(viz_plots, 'create_rmsf_by_aa_ss_density', None), [avg_feature_df], {}),
              (viz_plots.create_voxel_info_plot, [], {'config': self.config, 'voxel_output_file': voxel_output}),
          ]
          # Filter out tasks where the function is None (e.g., if new plots aren't defined)
          plotting_tasks = [task for task in plotting_tasks if task[0] is not None]


          # --- Execute Plotting Tasks ---
          logging.info(f"Executing {len(plotting_tasks)} visualization tasks...")
          for plot_func, data_args, plot_kwargs in tqdm(plotting_tasks, desc="Generating Visualizations", disable=not self.config.get('logging',{}).get('show_progress_bars', True)):
                self.run_plot(plot_func, *data_args, **plot_kwargs)

          logging.info("Visualization generation attempt complete.")