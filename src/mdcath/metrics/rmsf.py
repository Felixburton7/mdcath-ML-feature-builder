"""
Module for processing Root Mean Square Fluctuation (RMSF) data.
Includes extraction, validation, averaging, and saving.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from mdcath.io.hdf5_reader import HDF5Reader
from mdcath.io.writers import save_dataframe_csv, get_rmsf_output_path

class RmsfProcessingError(Exception):
    """Custom exception for RMSF processing errors."""
    pass

def process_domain_rmsf(domain_id: str,
                        reader: HDF5Reader,
                        config: Dict[str, Any]
                        ) -> Optional[Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Extracts RMSF data for all temps/replicas and aligns with residue info.
    Standardizes Histidine variants to 'HIS'.

    Args:
        domain_id (str): The domain ID.
        reader (HDF5Reader): Initialized HDF5Reader for the domain.
        config (Dict[str, Any]): Global configuration.

    Returns:
        Optional[Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]]:
            - DataFrame with unique 'resid', 'resname' (standardized).
            - Dict[temp_str, Dict[rep_str, np.ndarray]] holding VALIDATED RMSF arrays.
            Returns None if validation fails or no RMSF data found.
    """
    logging.debug(f"Processing RMSF for domain {domain_id}")
    residue_info_df = reader.get_residue_info() # This function already handles decoding bytes
    if residue_info_df is None or residue_info_df.empty:
        logging.error(f"Cannot process RMSF for {domain_id}: Missing or empty residue info.")
        return None

    # *** STANDARDIZE HISTIDINE NAMES HERE ***
    his_variants = ['HSD', 'HSE', 'HSP', 'HID', 'HIE', 'HIP'] # Include common variants
    original_resnames = set(residue_info_df['resname'].unique())
    variants_found = [res for res in his_variants if res in original_resnames]

    if variants_found:
        logging.debug(f"Standardizing Histidine variants {variants_found} to HIS for domain {domain_id}")
        # Create a mapping dictionary
        his_map = {variant: 'HIS' for variant in his_variants}
        # Apply the replacement
        residue_info_df['resname'] = residue_info_df['resname'].replace(his_map)
        standardized_resnames = set(residue_info_df['resname'].unique())
        logging.debug(f"Resnames after standardization for {domain_id}: {standardized_resnames}")
    # --- End Histidine Standardization ---

    num_residues = len(residue_info_df)
    rmsf_data_collected: Dict[str, Dict[str, np.ndarray]] = {}
    found_any_rmsf = False
    temps_to_process = reader.get_available_temperatures() # Use all available temps

    for temp in temps_to_process:
        temp_str = str(temp)
        rmsf_data_collected[temp_str] = {}
        replicas_to_process = reader.get_available_replicas(temp)

        for replica in replicas_to_process:
            replica_str = str(replica)
            rmsf_array = reader.get_rmsf(temp, replica)

            if rmsf_array is None:
                logging.warning(f"RMSF not found/readable for {domain_id}, T={temp_str}, R={replica_str}")
                continue

            # --- CRITICAL VALIDATION ---
            if len(rmsf_array) != num_residues:
                logging.error(f"RMSF length mismatch for {domain_id}, T={temp_str}, R={replica_str}: "
                              f"Expected {num_residues} residues, got {len(rmsf_array)} RMSF values. Skipping this replica.")
                continue # Skip this invalid replica

            # --- Check for NaNs ---
            if np.isnan(rmsf_array).any():
                 logging.warning(f"NaN values found in RMSF data for {domain_id}, T={temp_str}, R={replica_str}. "
                                 f"Attempting to fill with mean of valid values.")
                 if np.isnan(rmsf_array).all():
                      logging.error(f"All RMSF values are NaN for {domain_id}, T={temp_str}, R={replica_str}. Skipping.")
                      continue
                 mean_rmsf = np.nanmean(rmsf_array)
                 rmsf_array[np.isnan(rmsf_array)] = mean_rmsf

            rmsf_data_collected[temp_str][replica_str] = rmsf_array
            found_any_rmsf = True

    if not found_any_rmsf:
        logging.warning(f"No valid RMSF data found for any temp/replica for domain {domain_id}.")
        return None

    # Add domain_id column to residue info for easier aggregation later
    residue_info_df['domain_id'] = domain_id
    residue_info_df = residue_info_df[['domain_id', 'resid', 'resname']] # Reorder

    return residue_info_df, rmsf_data_collected


def aggregate_and_average_rmsf(all_domain_rmsf_results: Dict[str, Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]],
                               config: Dict[str, Any]
                               ) -> Tuple[Optional[Dict], Optional[Dict], Optional[pd.DataFrame]]:
    """
    Aggregates RMSF data across all domains, calculates averages, and prepares for saving.

    Args:
        all_domain_rmsf_results (Dict): Output from process_domain_rmsf for multiple domains.
                                        {domain_id: (residue_info_df, rmsf_data_dict)}
        config (Dict[str, Any]): Global configuration.

    Returns:
        Tuple[Optional[Dict], Optional[Dict], Optional[pd.DataFrame]]:
            - Aggregated raw replica data dict: {temp: {replica: combined_df}}
            - Aggregated replica average data dict: {temp: combined_avg_df}
            - Overall temperature average DataFrame across all domains.
            Returns (None, None, None) if input is empty or processing fails.
    """
    if not all_domain_rmsf_results:
        logging.warning("No domain RMSF results provided for aggregation.")
        return None, None, None

    logging.info(f"Aggregating RMSF data for {len(all_domain_rmsf_results)} domains...")

    # Structures to hold combined data
    combined_replica_dfs: Dict[str, Dict[str, List[pd.DataFrame]]] = {} # {temp: {replica: [df1, df2, ...]}}
    temps = set()

    # 1. Combine raw replica data from all domains
    for domain_id, (res_info_df, rmsf_dict) in all_domain_rmsf_results.items():
        if res_info_df is None or rmsf_dict is None: continue

        for temp_str, replica_dict in rmsf_dict.items():
            temps.add(temp_str)
            if temp_str not in combined_replica_dfs:
                combined_replica_dfs[temp_str] = {}

            for replica_str, rmsf_array in replica_dict.items():
                if replica_str not in combined_replica_dfs[temp_str]:
                    combined_replica_dfs[temp_str][replica_str] = []

                # Create DataFrame for this domain/temp/replica
                domain_temp_rep_df = res_info_df.copy()
                rmsf_col_name = f"rmsf_{temp_str}" # Consistent naming
                domain_temp_rep_df[rmsf_col_name] = rmsf_array
                combined_replica_dfs[temp_str][replica_str].append(domain_temp_rep_df)

    if not combined_replica_dfs:
        logging.error("Failed to combine any replica RMSF data.")
        return None, None, None

    # Concatenate DataFrames for each temp/replica
    final_replica_data: Dict[str, Dict[str, pd.DataFrame]] = {} # {temp: {replica: final_df}}
    for temp_str, replica_dict in combined_replica_dfs.items():
        final_replica_data[temp_str] = {}
        for replica_str, df_list in replica_dict.items():
            if df_list:
                 final_replica_data[temp_str][replica_str] = pd.concat(df_list, ignore_index=True)
                 logging.debug(f"Combined replica data for T={temp_str}, R={replica_str}: {len(final_replica_data[temp_str][replica_str])} rows")
            else:
                 logging.warning(f"No dataframes to combine for T={temp_str}, R={replica_str}")


    # 2. Calculate Replica Averages for each temperature
    replica_average_data: Dict[str, pd.DataFrame] = {} # {temp: final_avg_df}
    all_averages_list = [] # List to store average DFs per temp for final overall average

    for temp_str in sorted(list(temps)): # Process in consistent order
        if temp_str not in final_replica_data or not final_replica_data[temp_str]:
            logging.warning(f"No replica data found for T={temp_str} to calculate average.")
            continue

        replica_dfs_for_temp = list(final_replica_data[temp_str].values())
        if not replica_dfs_for_temp:
             logging.warning(f"DataFrame list is empty for T={temp_str} average calculation.")
             continue

        # Use the first replica's DF structure as base, group by domain and residue
        base_df_structure = replica_dfs_for_temp[0][['domain_id', 'resid', 'resname']] # Should be consistent
        all_replica_rmsf = [df[[f"rmsf_{temp_str}"]].rename(columns={f"rmsf_{temp_str}": f"rmsf_rep_{i}"})
                           for i, df in enumerate(replica_dfs_for_temp)]

        # Concatenate along columns (assuming rows align perfectly based on prior processing)
        # This relies heavily on the domain processing having consistent residue info
        # A safer approach might group by (domain_id, resid) but could be slower.
        # Let's assume concatenation works if indices are aligned after pd.concat above.
        # Need to verify index alignment or reset index before merge/concat.

        # Group by approach (safer but potentially slower)
        try:
            all_temp_reps = pd.concat(replica_dfs_for_temp, ignore_index=True)
            grouped = all_temp_reps.groupby(['domain_id', 'resid', 'resname'])
            # Calculate mean RMSF within each group
            avg_rmsf_series = grouped[f"rmsf_{temp_str}"].mean()
            # Convert back to DataFrame
            avg_df_temp = avg_rmsf_series.reset_index()
            # Rename column for consistency? Let's keep rmsf_{temp_str} for now.
            # avg_df_temp = avg_df_temp.rename(columns={f"rmsf_{temp_str}": f"rmsf_avg_{temp_str}"})

            if not avg_df_temp.empty:
                 replica_average_data[temp_str] = avg_df_temp
                 all_averages_list.append(avg_df_temp.rename(columns={f"rmsf_{temp_str}": f"rmsf_avg_at_{temp_str}"})) # Rename for merge
                 logging.info(f"Calculated replica average for T={temp_str}: {len(avg_df_temp)} rows")
            else:
                 logging.warning(f"Replica average calculation resulted in empty DataFrame for T={temp_str}")

        except Exception as e:
             logging.error(f"Failed to calculate replica average for T={temp_str}: {e}", exc_info=True)


    # 3. Calculate Overall Temperature Average
    overall_average_df = None
    if not all_averages_list:
        logging.error("No replica averages available to calculate overall temperature average.")
    else:
        try:
            # Start with the first temperature's average DF
            overall_average_df = all_averages_list[0][['domain_id', 'resid', 'resname', f"rmsf_avg_at_{sorted(list(temps))[0]}"]]
            # Iteratively merge other temperatures
            for i in range(1, len(all_averages_list)):
                temp_str = sorted(list(temps))[i]
                merge_df = all_averages_list[i][['domain_id', 'resid', f"rmsf_avg_at_{temp_str}"]]
                overall_average_df = pd.merge(overall_average_df, merge_df, on=['domain_id', 'resid'], how='outer')

            # Calculate the mean across all 'rmsf_avg_at_TEMP' columns
            rmsf_avg_cols = [col for col in overall_average_df.columns if col.startswith("rmsf_avg_at_")]
            if rmsf_avg_cols:
                 overall_average_df['rmsf_average'] = overall_average_df[rmsf_avg_cols].mean(axis=1, skipna=True)
                 # Optionally remove the per-temp columns after calculating overall average
                 # overall_average_df = overall_average_df.drop(columns=rmsf_avg_cols)
                 logging.info(f"Calculated overall temperature average: {len(overall_average_df)} rows")
            else:
                 logging.error("No average RMSF columns found to calculate overall average.")
                 overall_average_df = None # Reset if calculation failed

        except Exception as e:
            logging.error(f"Failed to calculate overall temperature average: {e}", exc_info=True)
            overall_average_df = None

    return final_replica_data, replica_average_data, overall_average_df


def save_rmsf_results(output_dir: str, config: Dict[str, Any],
                      replica_data: Optional[Dict],
                      replica_avg_data: Optional[Dict],
                      overall_avg_data: Optional[pd.DataFrame]):
    """
    Saves all processed RMSF data to CSV files according to configuration.

    Args:
        output_dir (str): Base output directory.
        config (Dict[str, Any]): Global configuration.
        replica_data (Optional[Dict]): Aggregated raw replica data.
        replica_avg_data (Optional[Dict]): Aggregated replica average data.
        overall_avg_data (Optional[pd.DataFrame]): Overall temperature average data.
    """
    flatten_dirs = config.get("output", {}).get("flatten_rmsf_dirs", True)
    rmsf_base = os.path.join(output_dir, "RMSF")

    # 1. Save Raw Replica Data
    if replica_data:
        logging.info("Saving raw replica RMSF data...")
        for temp_str, replica_dict in replica_data.items():
            for replica_str, df in replica_dict.items():
                 try:
                     path = get_rmsf_output_path(output_dir, 'replica', flatten_dirs,
                                                 temperature=temp_str, replica=replica_str)
                     save_dataframe_csv(df, path)
                 except Exception as e:
                      logging.error(f"Failed to save RMSF for T={temp_str}, R={replica_str}: {e}")
    else:
         logging.warning("No raw replica RMSF data to save.")

    # 2. Save Replica Average Data
    if replica_avg_data:
        logging.info("Saving replica average RMSF data...")
        for temp_str, df in replica_avg_data.items():
            try:
                path = get_rmsf_output_path(output_dir, 'average', flatten_dirs, temperature=temp_str)
                # Ensure column name is consistent as 'rmsf_{temp}' before saving
                df_to_save = df.rename(columns={f"rmsf_{temp_str}": f"rmsf_{temp_str}"}, errors='ignore') # No change needed if correct
                save_dataframe_csv(df_to_save, path)
            except Exception as e:
                logging.error(f"Failed to save replica average RMSF for T={temp_str}: {e}")
    else:
         logging.warning("No replica average RMSF data to save.")

    # 3. Save Overall Average Data
    if overall_avg_data is not None and not overall_avg_data.empty:
        logging.info("Saving overall temperature average RMSF data...")
        try:
             path = get_rmsf_output_path(output_dir, 'average', flatten_dirs) # No temp specified
             # Ensure the column is named 'rmsf_average'
             df_to_save = overall_avg_data[['domain_id', 'resid', 'resname', 'rmsf_average']].copy()
             save_dataframe_csv(df_to_save, path)
        except KeyError:
             logging.error(f"Column 'rmsf_average' not found in overall average DataFrame. Cannot save.")
        except Exception as e:
             logging.error(f"Failed to save overall average RMSF: {e}")
    else:
         logging.warning("No overall average RMSF data to save.")

