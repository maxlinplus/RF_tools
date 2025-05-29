import pandas as pd
import streamlit as st
import sys
import os
import re
import io
import json # Import json for saving/loading settings

# Define custom sort order for LTE Bands
LTE_BAND_ORDER = [
    "Band1", "Band2", "Band3", "Band4", "Band5", "Band7", "Band8", "Band12",
    "Band13", "Band17", "Band18", "Band19", "Band20", "Band25", "Band26",
    "Band28", "Band66", "Band71"
]

# Helper function: Sort bands according to LTE_BAND_ORDER, placing unknown bands at the end
def sort_bands(bands):
    def get_sort_key(band):
        try:
            return LTE_BAND_ORDER.index(band)
        except ValueError:
            return len(LTE_BAND_ORDER) # Place unknown bands at the end
    return sorted(bands, key=get_sort_key)

# Define MPR backoff rules
# Key: (BW, Description) -> Value: target power backoff
MPR_BACKOFF_RULES = {
    ('1.4 MHz', 'QPSK, 5 (RB_Pos:LOW)'): 0,
    ('1.4 MHz', 'QPSK, 5 (RB_Pos:HIGH)'): 0,
    ('1.4 MHz', 'QPSK, 6 (RB_Pos:LOW)'): 1,
    ('1.4 MHz', 'Q16, 5 (RB_Pos:LOW)'): 1,
    ('1.4 MHz', 'Q16, 5 (RB_Pos:HIGH)'): 1,
    ('1.4 MHz', 'Q16, 6 (RB_Pos:LOW)'): 2,
    ('3.0 MHz', 'QPSK, 4 (RB_Pos:LOW)'): 0,
    ('3.0 MHz', 'QPSK, 4 (RB_Pos:HIGH)'): 0,
    ('3.0 MHz', 'QPSK, 15 (RB_Pos:LOW)'): 1,
    ('3.0 MHz', 'Q16, 4 (RB_Pos:LOW)'): 1,
    ('3.0 MHz', 'Q16, 4 (RB_Pos:HIGH)'): 1,
    ('3.0 MHz', 'Q16, 15 (RB_Pos:LOW)'): 2,
    ('5.0 MHz', 'QPSK, 8 (RB_Pos:LOW)'): 0,
    ('5.0 MHz', 'QPSK, 8 (RB_Pos:HIGH)'): 0,
    ('5.0 MHz', 'QPSK, 25 (RB_Pos:LOW)'): 1,
    ('5.0 MHz', 'Q16, 8 (RB_Pos:LOW)'): 1,
    ('5.0 MHz', 'Q16, 8 (RB_Pos:HIGH)'): 1,
    ('5.0 MHz', 'Q16, 25 (RB_Pos:LOW)'): 2,
    ('10 MHz', 'QPSK, 12 (RB_Pos:LOW)'): 0,
    ('10 MHz', 'QPSK, 12 (RB_Pos:HIGH)'): 0,
    ('10 MHz', 'QPSK, 50 (RB_Pos:LOW)'): 1,
    ('10 MHz', 'Q16, 12 (RB_Pos:LOW)'): 1,
    ('10 MHz', 'Q16, 12 (RB_Pos:HIGH)'): 1,
    ('10 MHz', 'Q16, 50 (RB_Pos:LOW)'): 2,
    ('15 MHz', 'QPSK, 16 (RB_Pos:LOW)'): 0,
    ('15 MHz', 'QPSK, 16 (RB_Pos:HIGH)'): 0,
    ('15 MHz', 'QPSK, 75 (RB_Pos:LOW)'): 1,
    ('15 MHz', 'Q16, 16 (RB_Pos:LOW)'): 1,
    ('15 MHz', 'Q16, 16 (RB_Pos:HIGH)'): 1,
    ('15 MHz', 'Q16, 75 (RB_Pos:LOW)'): 2,
    ('20 MHz', 'QPSK, 18 (RB_Pos:LOW)'): 0,
    ('20 MHz', 'QPSK, 18 (RB_Pos:HIGH)'): 0,
    ('20 MHz', 'QPSK, 100 (RB_Pos:LOW)'): 1,
    ('20 MHz', 'Q16, 18 (RB_Pos:LOW)'): 1,
    ('20 MHz', 'Q16, 18 (RB_Pos:HIGH)'): 1,
    ('20 MHz', 'Q16, 100 (RB_Pos:LOW)'): 2,
}


# ====================================================================
# Core Analysis Logic (analyze_lte_report function)
# ====================================================================
def analyze_lte_report(df, band_target_powers_dict, power_tolerance, evm_margin_threshold, sem_margin_threshold, aclr_margin_threshold, mpr_tolerance, progress_callback=None):
    """
    Analyzes RF LTE validation report and identifies test items requiring attention.
   
    Args:
        df (pd.DataFrame): The DataFrame containing the report data.
        band_target_powers_dict (dict): Dictionary of target powers for each Band.
        power_tolerance (float): Power tolerance for 6.2.2 Maximum Output Power.
        evm_margin_threshold (float): Margin alert threshold for 6.5.2.1 EVM.
        sem_margin_threshold (float): Margin alert threshold for 6.6.2.1 SEM.
        aclr_margin_threshold (float): Margin alert threshold for 6.6.2.3 ACLR.
        mpr_tolerance (float): Tolerance for 6.2.3 Maximum Power Reduction.
        progress_callback (callable, optional): A function to call with progress updates (0-100).
    Returns:
        tuple: A tuple containing (dict: attention_items, set: found_categories_in_report, int: total_test_items_checked, list: original_columns, list: bands_checked).
    """
    if progress_callback:
        progress_callback(5)
    
    # Store original columns to maintain report format
    original_columns = df.columns.tolist()
    # Remove 'Unit' column if it exists in original columns
    if 'Unit' in original_columns:
        original_columns.remove('Unit')
    # Remove 'Expected Backoff' column if it exists in original columns
    if 'Expected Backoff' in original_columns:
        original_columns.remove('Expected Backoff')

    # Get unique bands found in the report for summary
    bands_checked = []
    if 'Band' in df.columns:
        bands_checked = sorted(df['Band'].astype(str).str.strip().unique().tolist(), key=lambda x: LTE_BAND_ORDER.index(x) if x in LTE_BAND_ORDER else len(LTE_BAND_ORDER))


    found_categories_in_report = set(df['Test Item Type'].astype(str).str.strip().unique())
    total_test_items_checked = 0

    attention_items = {
        "6.2.2 Maximum Output Power": [],
        "6.2.3 Maximum Power Reduction": [],
        "6.5.2.1 Error Vector Magnitude (EVM) for PUSCH": [],
        "6.6.2.1 Spectrum Emission Mask": [],
        "6.6.2.3 Adjacent Channel Leakage Power Ratio": []
    }

    # Helper function to prepare a row for attention_items, including all original columns
    def prepare_attention_row(original_row_series, category_key, message, extra_fields=None):
        row_dict = original_row_series.to_dict()
        row_dict['Attention_Message'] = message # Add a specific message for attention
        # Remove 'Unit' if it exists in the row_dict
        if 'Unit' in row_dict:
            del row_dict['Unit']
        # Remove 'Expected Backoff' if it exists
        if 'Expected Backoff' in row_dict:
            del row_dict['Expected Backoff']
        # Remove 'Limit Low', 'Limit High', 'Target Range', 'Threshold' if they exist, as they are now replaced by 'Criteria'
        for col_to_remove in ['Limit Low', 'Limit High', 'Target Range', 'Threshold']:
            if col_to_remove in row_dict:
                del row_dict[col_to_remove]

        if extra_fields:
            row_dict.update(extra_fields)
        return row_dict


    total_rows = len(df)
    processed_rows = 0

    # --- 6.2.2 Maximum Output Power ---
    category_key = "6.2.2 Maximum Output Power"
    target_rows = df[df['Test Item Type'].astype(str).str.strip() == category_key]
    total_test_items_checked += len(target_rows)
    for index, row in target_rows.iterrows():
        try:
            measured_power = pd.to_numeric(row['Measured'], errors='coerce')
            current_band = str(row.get('Band', '')).strip()
            # status = str(row.get('Status', '')).strip() # Not used for attention check in this section

            current_band_target_power = band_target_powers_dict.get(current_band, 23.0)

            extra_fields = {'Margin': 'N/A', 'Criteria': 'N/A'}

            if pd.isna(measured_power):
                attention_items[category_key].append(
                    prepare_attention_row(row, category_key,
                                          f"Measured value could not be parsed as a number: '{row.get('Measured', '')}'",
                                          extra_fields)
                )
                continue

            lower_bound = current_band_target_power - power_tolerance
            upper_bound = current_band_target_power + power_tolerance
            
            extra_fields['Criteria'] = f"[{lower_bound:.2f}, {upper_bound:.2f}]"
            extra_fields['Measured'] = f"{measured_power:.2f}" # Format measured for consistency

            if not (lower_bound <= measured_power <= upper_bound):
                attention_items[category_key].append(
                    prepare_attention_row(row, category_key,
                                          f"Measured: {measured_power:.2f} dBm, target range [{lower_bound:.2f}, {upper_bound:.2f}] dBm",
                                          extra_fields)
                )
        except Exception as e:
            # print(f"Error processing 6.2.2 row: {e} - Row: {row.to_dict()}")
            pass # Silently pass errors for robust processing
        processed_rows += 1
        if progress_callback:
            progress_callback(15 + int(processed_rows / total_rows * 15))

    # --- 6.2.3 Maximum Power Reduction ---
    category_key = "6.2.3 Maximum Power Reduction"
    target_rows = df[df['Test Item Type'].astype(str).str.strip() == category_key]
    total_test_items_checked += len(target_rows)
    for index, row in target_rows.iterrows():
        try:
            measured_val = pd.to_numeric(row.get('Measured', ''), errors='coerce')
            bw = str(row.get('BW', '')).strip()
            description = str(row.get('Description', '')).strip()
            current_band = str(row.get('Band', '')).strip()
            # status = str(row.get('Status', '')).strip() # Not used for attention check in this section

            current_band_target_power = band_target_powers_dict.get(current_band, 23.0)
            rule_key = (bw, description)
            expected_backoff = MPR_BACKOFF_RULES.get(rule_key)

            extra_fields = {'Margin': 'N/A', 'Criteria': 'N/A'}
            message = ""

            if pd.isna(measured_val):
                message = "Measured value could not be parsed as a number"
                attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
                continue

            if expected_backoff is None:
                message = f"No MPR backoff rule found for BW: '{bw}', Description: '{description}'"
                attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
                continue

            calculated_target_power = current_band_target_power - expected_backoff
            lower_bound_mpr = calculated_target_power - mpr_tolerance
            upper_bound_mpr = calculated_target_power + mpr_tolerance

            extra_fields["Criteria"] = f"[{lower_bound_mpr:.2f}, {upper_bound_mpr:.2f}]"
            extra_fields["Measured"] = f"{measured_val:.2f}"

            if not (lower_bound_mpr <= measured_val <= upper_bound_mpr):
                 message = f"Measured: {measured_val:.2f} dBm, calculated range [{lower_bound_mpr:.2f}, {upper_bound_mpr:.2f}] dBm. Expected Backoff: {expected_backoff:.1f} dB"
                 attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
        except Exception as e:
            # print(f"Error processing 6.2.3 row: {e} - Row: {row.to_dict()}")
            pass
        processed_rows += 1
        if progress_callback:
            progress_callback(30 + int(processed_rows / total_rows * 15))


    # --- 6.5.2.1 Error Vector Magnitude (EVM) for PUSCH ---
    category_key = "6.5.2.1 Error Vector Magnitude (EVM) for PUSCH"

    target_rows = df[df['Test Item Type'].astype(str).str.strip() == category_key]
    total_test_items_checked += len(target_rows)

    for index, row in target_rows.iterrows():
        try:
            measured_val = pd.to_numeric(row['Measured'], errors='coerce')
            limit_high = pd.to_numeric(row.get('Limit High', ''), errors='coerce')
            # status = str(row.get('Status', '')).strip() # Not used for attention check in this section

            extra_fields = {"Margin": "N/A", "Criteria": "N/A"}
            message = ""

            if pd.isna(measured_val) or pd.isna(limit_high):
                message = "Measured value or upper limit could not be parsed as numbers."
                attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
                continue

            calculated_margin = limit_high - measured_val
            extra_fields["Margin"] = f"{calculated_margin:.2f}"
            extra_fields["Criteria"] = f"{evm_margin_threshold:.1f}" # Only show threshold number
            extra_fields["Measured"] = f"{measured_val:.2f}"

            if calculated_margin < evm_margin_threshold:
                message = f"Measured: {measured_val:.2f}, Upper Limit: {limit_high:.2f}. Margin: {calculated_margin:.2f}, Threshold {evm_margin_threshold}"
                attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
        except Exception as e:
            # print(f"Error processing 6.5.2.1 row: {e} - Row: {row.to_dict()}")
            pass
        processed_rows += 1
        if progress_callback:
            progress_callback(45 + int(processed_rows / total_rows * 20))


    # --- 6.6.2.1 Spectrum Emission Mask ---
    category_key = "6.6.2.1 Spectrum Emission Mask"

    target_rows = df[df['Test Item Type'].astype(str).str.strip() == category_key]
    total_test_items_checked += len(target_rows)

    for index, row in target_rows.iterrows():
        try:
            measured_val = pd.to_numeric(row['Measured'], errors='coerce')
            limit_low_val = pd.to_numeric(row.get('Limit Low', ''), errors='coerce')
            limit_high_str = str(row.get('Limit High', '')).strip()
            # status = str(row.get('Status', '')).strip() # Not used for attention check in this section

            extra_fields = {"Margin": "N/A", "Criteria": "N/A"}
            message = ""

            if pd.isna(measured_val):
                message = "SEM measured value could not be parsed as a number."
                attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
                continue

            calculated_margin = None
            
            if limit_high_str == '---': # Lower limit only
                if pd.notna(limit_low_val):
                    calculated_margin = measured_val - limit_low_val
                    extra_fields['Criteria'] = f"{sem_margin_threshold:.1f}" # Show threshold only
                else:
                    message = "Cannot calculate Margin (Invalid Limit Low/High)."
                    attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
                    continue
            else: # Both limits or upper limit only
                limit_high_val = pd.to_numeric(limit_high_str, errors='coerce')
                if pd.notna(limit_high_val):
                    calculated_margin_high = limit_high_val - measured_val
                    if pd.notna(limit_low_val):
                        calculated_margin_low = measured_val - limit_low_val
                        calculated_margin = min(calculated_margin_high, calculated_margin_low)
                    else: # Only upper limit (if lower is NaN or not present)
                        calculated_margin = calculated_margin_high
                    extra_fields['Criteria'] = f"{sem_margin_threshold:.1f}" # Show threshold only
                else:
                    message = "Cannot calculate Margin (Invalid Limit High)."
                    attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
                    continue

            if calculated_margin is not None:
                extra_fields["Margin"] = f"{calculated_margin:.2f}"
                extra_fields["Measured"] = f"{measured_val:.2f}"


            if calculated_margin is not None and calculated_margin < sem_margin_threshold:
                message = f"Measured: {measured_val:.2f} dB. Margin: {calculated_margin:.2f} dB, Threshold {sem_margin_threshold} dB"
                attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
        except Exception as e:
            # print(f"Error processing 6.6.2.1 row: {e} - Row: {row.to_dict()}")
            pass
        processed_rows += 1
        if progress_callback:
            progress_callback(65 + int(processed_rows / total_rows * 20))


    # --- 6.6.2.3 Adjacent Channel Leakage Power Ratio ---
    category_key = "6.6.2.3 Adjacent Channel Leakage Power Ratio"

    target_rows = df[df['Test Item Type'].astype(str).str.strip() == category_key]

    # total_test_items_checked logic for ACLR:
    # We need to explicitly count only the ACLR items that are not 'UE Maximum Output Power' sub-items
    # as these are already counted under 6.2.2.
    for index, row in target_rows.iterrows():
        test_item_name = str(row.get('Test Item Name', '')).strip()
        if 'UE Maximum Output Power' not in test_item_name:
            total_test_items_checked += 1 # Count only relevant ACLR items

    for index, row in target_rows.iterrows():
        try:
            test_item_name = str(row.get('Test Item Name', '')).strip()
            if 'UE Maximum Output Power' in test_item_name:
                continue

            measured_val = pd.to_numeric(row['Measured'], errors='coerce')
            limit_low_val = pd.to_numeric(row.get('Limit Low', ''), errors='coerce')
            limit_high_str = str(row.get('Limit High', '')).strip()
            # status = str(row.get('Status', '')).strip() # Not used for attention check in this section

            extra_fields = {"Margin": "N/A", "Criteria": "N/A"}
            message = ""

            if pd.isna(measured_val):
                message = "ACLR measured value could not be parsed as a number."
                attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
                continue

            calculated_margin = None

            if limit_high_str == '---': # Lower limit only
                if pd.notna(limit_low_val):
                    calculated_margin = measured_val - limit_low_val
                    extra_fields['Criteria'] = f"{aclr_margin_threshold:.1f}" # Show threshold only
                else:
                    message = "Cannot calculate Margin (Invalid Limit Low/High)."
                    attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
                    continue
            else: # Both limits or upper limit only
                limit_high_val = pd.to_numeric(limit_high_str, errors='coerce')
                if pd.notna(limit_high_val):
                    calculated_margin_high = limit_high_val - measured_val
                    if pd.notna(limit_low_val):
                        calculated_margin_low = measured_val - limit_low_val
                        calculated_margin = min(calculated_margin_high, calculated_margin_low)
                    else: # Only upper limit (if lower is NaN or not present)
                        calculated_margin = calculated_margin_high
                    extra_fields['Criteria'] = f"{aclr_margin_threshold:.1f}" # Show threshold only
                else:
                    message = "Cannot calculate Margin (Invalid Limit High)."
                    attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
                    continue

            if calculated_margin is not None:
                extra_fields["Margin"] = f"{calculated_margin:.2f}"
                extra_fields["Measured"] = f"{measured_val:.2f}"

            if calculated_margin is not None and calculated_margin < aclr_margin_threshold:
                message = f"Measured: {measured_val:.2f} dB. Margin: {calculated_margin:.2f} dB, Threshold {aclr_margin_threshold} dB"
                attention_items[category_key].append(prepare_attention_row(row, category_key, message, extra_fields))
        except Exception as e:
            # print(f"Error processing 6.6.2.3 row: {e} - Row: {row.to_dict()}")
            pass
        processed_rows += 1
        if progress_callback:
            progress_callback(85 + int(processed_rows / total_rows * 10))

    if progress_callback:
        progress_callback(100)

    # Return original_columns as well to the GUI for dynamic Treeview setup
    return attention_items, found_categories_in_report, total_test_items_checked, original_columns, bands_checked


# ====================================================================
# CSV Conversion Logic (from CMWrun_csv_data_convert_V1.py) - Integrated
# ====================================================================
def convert_raw_csv_to_dataframe(uploaded_file, progress_callback=None): # 接受檔案物件
    """
    Reads a raw CSV file, performs data conversion and filtering,
    and returns a pandas DataFrame.
    """
    if progress_callback:
        progress_callback(5) # Start conversion progress
    
    try:
        # 直接使用 BytesIO 讀取上傳的檔案內容
        # 嘗試 UTF-8 解碼，如果失敗則嘗試 latin-1
        try:
            raw_data_lines = uploaded_file.readlines()
            # 確保檔案指標重置到開頭
            uploaded_file.seek(0) 
            raw_data_lines = [line.decode('utf-8') for line in raw_data_lines]
        except UnicodeDecodeError:
            uploaded_file.seek(0) # 重置檔案指標
            raw_data_lines = uploaded_file.readlines()
            raw_data_lines = [line.decode('latin-1') for line in raw_data_lines]
        
        processed_rows = []
        
        current_test_item_type = ""
        current_test_item_band = ""

        # === List of test item types to filter ===
        filter_test_types = [
            "6.2.2 Maximum Output Power",
            "6.2.3 Maximum Power Reduction",
            "6.5.2.1 Error Vector Magnitude (EVM) for PUSCH",
            "6.6.2.1 Spectrum Emission Mask",
            "6.6.2.3 Adjacent Channel Leakage Power Ratio"
        ]
        # =======================================

        headers = [
            "Test Item Type",
            "Band",
            "Test Item Name",
            "ULCH",
            "BW",
            "Description",
            "Limit Low",
            "Limit High",
            "Measured",
            "Unit",
            "Status"
        ]

        total_lines = len(raw_data_lines)
        for i, line in enumerate(raw_data_lines):
            if progress_callback and (i % 100 == 0 or i == total_lines - 1): # Update infrequently for performance
                progress_callback(5 + int((i / total_lines) * 90)) # Scale progress for conversion

            parts = [p.strip().replace('"', '') for p in line.split(';')]

            if not parts or parts[0].strip() == '':
                continue

            row_type = parts[0].strip()
            
            current_row_data = {header: "" for header in headers}

            if row_type == "TestItemList":
                if len(parts) > 1:
                    test_list_str = parts[1].strip()
                    match_band = re.search(r'@\s*(Band\d+)', test_list_str, re.IGNORECASE)
                    if match_band:
                        current_test_item_band = match_band.group(1)
                        test_item_type_temp = test_list_str.replace(match_band.group(0), '').strip()
                        current_test_item_type = test_item_type_temp.rstrip(':').strip()
                    else:
                        current_test_item_type = test_list_str
                        current_test_item_band = ""

            elif row_type == "TestItem":
                if filter_test_types and current_test_item_type not in filter_test_types:
                    continue

                if len(parts) >= 9:
                    current_row_data["Test Item Type"] = current_test_item_type
                    current_row_data["Band"] = current_test_item_band

                    # Clean Test Item Name: remove numbers like '1.' and '2.' at the beginning
                    test_item_name_raw = parts[1]
                    clean_test_item_name = re.sub(r'^\d+\.\s*', '', test_item_name_raw).strip()
                    current_row_data["Test Item Name"] = clean_test_item_name


                    ulch_bw_context = parts[2].strip()
                    match_ulch = re.search(r'ULCH:\s*(\d+)', ulch_bw_context, re.IGNORECASE)
                    if match_ulch:
                        current_row_data["ULCH"] = match_ulch.group(1).strip()
                    match_bw = re.search(r'BW:\s*([\d\.]+\s*MHz)', ulch_bw_context, re.IGNORECASE)
                    if match_bw:
                        current_row_data["BW"] = match_bw.group(1).strip()
                    
                    ul_mod_rb_context = parts[3].strip()
                    
                    temp_ul_mod = ""
                    temp_ul_rb = ""
                    temp_rb_pos = ""

                    match_ul_mod_rb_pos_final = re.search(
                        r'UL_MOD_RB:\s*([A-Za-z0-9]+)'
                        r'(?:\t|\s+)(\d+)?'
                        r'\s*\((?:RB_Pos:)?(.+?)\)?$',
                        ul_mod_rb_context, re.IGNORECASE
                    )

                    if match_ul_mod_rb_pos_final:
                        temp_ul_mod = match_ul_mod_rb_pos_final.group(1).strip() if match_ul_mod_rb_pos_final.group(1) else ""
                        temp_ul_rb = match_ul_mod_rb_pos_final.group(2).strip() if match_ul_mod_rb_pos_final.group(2) else ""
                        temp_rb_pos = match_ul_mod_rb_pos_final.group(3).strip() if match_ul_mod_rb_pos_final.group(3) else ""
                    else:
                        temp_parts = ul_mod_rb_context.replace('UL_MOD_RB:', '').strip().split('\t', 1)
                        temp_ul_mod = temp_parts[0] if temp_parts else ""
                        if len(temp_parts) > 1:
                            rb_pos_match = re.search(r'(?:\(RB_Pos:)?(.+?)\)?', temp_parts[1])
                            if rb_pos_match:
                                temp_rb_pos = rb_pos_match.group(1).strip()
                            temp_ul_rb = re.sub(r'\s*\(RB_Pos:.*?\)', '', temp_parts[1]).strip()


                    description_parts = []
                    if temp_ul_mod:
                        description_parts.append(temp_ul_mod)
                    if temp_ul_rb:
                        description_parts.append(temp_ul_rb)
                    if temp_rb_pos:
                        description_parts.append(f"(RB_Pos:{temp_rb_pos})")
                    current_row_data["Description"] = " ".join(description_parts)

                    current_row_data["Limit Low"] = parts[4] if len(parts) > 4 else ""
                    current_row_data["Limit High"] = parts[5] if len(parts) > 5 else ""
                    current_row_data["Measured"] = parts[6] if len(parts) > 6 else ""
                    current_row_data["Unit"] = parts[7] if len(parts) > 7 else ""
                    current_row_data["Status"] = parts[8] if len(parts) > 8 else ""
                    
                    processed_rows.append(current_row_data)

        if not processed_rows:
            st.info("No convertible data rows found matching the filter criteria.") # 找不到符合篩選條件的可轉換資料列。
            return None

        df = pd.DataFrame(processed_rows, columns=headers)
        return df

    except Exception as e:
        st.error(f"An unexpected error occurred during conversion: {e}") # 轉換過程中發生未預期的錯誤
        return None

# ====================================================================
# Streamlit App Layout and Logic
# ====================================================================

# 配置 Streamlit 頁面
st.set_page_config(
    page_title="RF LTE Validation Report Analyzer Tool", # RF LTE 驗證報告分析工具
    layout="wide", # 讓頁面內容寬一點
    initial_sidebar_state="expanded" # 預設展開側邊欄
)

st.title("RF LTE Validation Report Analyzer Tool by lmax@") # RF LTE 驗證報告分析工具 by lmax@

# Function to load settings
def load_settings_from_file():
    uploaded_settings_file = st.session_state.get('uploaded_settings_file_sidebar')
    if uploaded_settings_file is not None:
        try:
            settings_data = json.load(uploaded_settings_file)
            st.session_state.power_tolerance = settings_data.get('power_tolerance', 1.0)
            st.session_state.mpr_tolerance = settings_data.get('mpr_tolerance', 1.0)
            st.session_state.evm_threshold = settings_data.get('evm_threshold', 5.0)
            st.session_state.sem_threshold = settings_data.get('sem_threshold', 5.0)
            st.session_state.aclr_threshold = settings_data.get('aclr_threshold', 3.0)
            
            loaded_band_powers = settings_data.get('band_target_powers_state', {})
            # Merge loaded settings with existing ones, ensuring all default bands are present
            # We must clear and then update to ensure old bands are removed if not in new settings
            st.session_state.band_target_powers_state = {band: 23.0 for band in LTE_BAND_ORDER} # Reset to default
            st.session_state.band_target_powers_state.update(loaded_band_powers) # Update with loaded values

            st.success("Settings loaded successfully! Please check the updated values in the sidebar.") # 設定已成功載入！請檢查側邊欄中更新的值。
        except Exception as e:
            st.error(f"Error loading settings: {e}") # 載入設定時發生錯誤

# 初始化 session_state 中的設定值
if 'power_tolerance' not in st.session_state:
    st.session_state.power_tolerance = 1.0
if 'mpr_tolerance' not in st.session_state:
    st.session_state.mpr_tolerance = 1.0
if 'evm_threshold' not in st.session_state:
    st.session_state.evm_threshold = 5.0
if 'sem_threshold' not in st.session_state:
    st.session_state.sem_threshold = 5.0
if 'aclr_threshold' not in st.session_state:
    st.session_state.aclr_threshold = 3.0
if 'band_target_powers_state' not in st.session_state:
    st.session_state.band_target_powers_state = {band: 23.0 for band in LTE_BAND_ORDER}

# 側邊欄用於設定參數
with st.sidebar:
    st.header("Analysis Settings") # 分析設定

    # Load/Save Settings Section
    st.subheader("Load/Save Settings") # 載入/儲存設定
    st.file_uploader("Upload Settings File (.json)", type=["json"], key="uploaded_settings_file_sidebar", on_change=load_settings_from_file) # 上傳設定檔 (.json)
    
    settings = {
        'power_tolerance': st.session_state.power_tolerance,
        'mpr_tolerance': st.session_state.mpr_tolerance,
        'evm_threshold': st.session_state.evm_threshold,
        'sem_threshold': st.session_state.sem_threshold,
        'aclr_threshold': st.session_state.aclr_threshold,
        'band_target_powers_state': st.session_state.band_target_powers_state
    }
    settings_json = json.dumps(settings, indent=4) 

    st.download_button(
        label="Save Current Settings", # 儲存目前設定
        data=settings_json,
        file_name="lte_analyzer_settings.json",
        mime="application/json",
        key="save_settings_button" 
    )

    # 通用閾值設定
    st.subheader("General Thresholds (Numerical)") # 通用閾值設定 (數值型)
    st.session_state.power_tolerance = st.number_input( # 改為 number_input
        "6.2.2 Max Output Power (±dB)", 
        min_value=0.1, max_value=5.0, 
        value=st.session_state.power_tolerance, step=0.1, format="%.1f",
        key="input_power_tolerance" # 更改 key 以避免與舊 slider 衝突
    )
    st.session_state.mpr_tolerance = st.number_input( # 改為 number_input
        "6.2.3 MPR Tolerance (±dB)", 
        min_value=0.1, max_value=5.0, 
        value=st.session_state.mpr_tolerance, step=0.1, format="%.1f",
        key="input_mpr_tolerance"
    )
    st.session_state.evm_threshold = st.number_input( # 改為 number_input
        "6.5.2.1 EVM Margin (<)", 
        min_value=1.0, max_value=10.0, 
        value=st.session_state.evm_threshold, step=0.5, format="%.1f",
        key="input_evm_threshold"
    )
    st.session_state.sem_threshold = st.number_input( # 改為 number_input
        "6.6.2.1 SEM Margin (<)", 
        min_value=1.0, max_value=10.0, 
        value=st.session_state.sem_threshold, step=0.5, format="%.1f",
        key="input_sem_threshold"
    )
    st.session_state.aclr_threshold = st.number_input( # 改為 number_input
        "6.6.2.3 ACLR Margin (<)", 
        min_value=0.1, max_value=10.0, 
        value=st.session_state.aclr_threshold, step=0.1, format="%.1f",
        key="input_aclr_threshold"
    )

    # Band 目標功率設定
    st.subheader("Band Target Power (dBm)") # Band 目標功率設定 (dBm)
    band_target_powers = {} 

    all_bands_in_state = set(st.session_state.band_target_powers_state.keys())
    bands_from_file = set(st.session_state.get('unique_bands_in_file', []))
    
    for band_in_file in bands_from_file:
        if band_in_file not in st.session_state.band_target_powers_state:
            st.session_state.band_target_powers_state[band_in_file] = 23.0

    bands_to_iterate_for_display = sort_bands(list(set(LTE_BAND_ORDER) | bands_from_file | all_bands_in_state))
    
    for band in bands_to_iterate_for_display:
        if band not in st.session_state.band_target_powers_state:
             st.session_state.band_target_powers_state[band] = 23.0

        current_band_power_value = float(st.session_state.band_target_powers_state.get(band, 23.0))
        
        st.session_state.band_target_powers_state[band] = st.number_input(
            f"{band} Target Power:",
            value=current_band_power_value, 
            step=0.1,      
            format="%.1f", 
            key=f"band_power_input_{band}"
        )
        band_target_powers[band] = st.session_state.band_target_powers_state[band]


# 主內容區塊
st.header("1. Upload Raw CSV Report") # 1. 上傳原始 CSV 報告
uploaded_file = st.file_uploader("Upload your raw CSV report file here (e.g., from CMWrun)", type=["csv"]) # 在此處上傳您的原始 CSV 報告檔案 (例如，來自 CMWrun)

# 使用 session_state 來管理文件上傳和轉換狀態
if 'uploaded_file_id' not in st.session_state:
    st.session_state.uploaded_file_id = None
if 'df_converted' not in st.session_state:
    st.session_state.df_converted = None
if 'converted_excel_bytes' not in st.session_state:
    st.session_state.converted_excel_bytes = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'unique_bands_in_file' not in st.session_state: 
    st.session_state.unique_bands_in_file = []


if uploaded_file is not None:
    if st.session_state.uploaded_file_id != uploaded_file.file_id:
        st.session_state.uploaded_file_id = uploaded_file.file_id
        st.session_state.df_converted = None 
        st.session_state.converted_excel_bytes = None
        st.session_state.analysis_results = None
        st.session_state.analysis_done = False
        st.session_state.unique_bands_in_file = [] 
        st.rerun() 

    if st.session_state.df_converted is None: 
        st.info("Converting raw CSV to a standardized format, please wait...") # 正在將原始 CSV 轉換為標準化格式，請稍候...
        
        conversion_progress_bar = st.progress(0)
        conversion_status_text = st.empty()

        def update_conversion_progress(value):
            conversion_progress_bar.progress(value / 100)
            conversion_status_text.text(f"Conversion Progress: {value}%") # 轉換進度

        df_converted_temp = convert_raw_csv_to_dataframe(uploaded_file, update_conversion_progress)
        
        if df_converted_temp is not None:
            st.success("Conversion Complete!") # 轉換完成！
            conversion_status_text.text("Conversion Complete.") # 轉換完成。
            st.write("Converted Data Preview (first 5 rows):") # 轉換後資料預覽 (前 5 列)：
            st.dataframe(df_converted_temp.head())
            
            st.session_state.df_converted = df_converted_temp

            output_converted_excel = io.BytesIO()
            with pd.ExcelWriter(output_converted_excel, engine='xlsxwriter') as writer:
                df_converted_temp.to_excel(writer, index=False, sheet_name='Converted Report')
            output_converted_excel.seek(0)
            st.session_state.converted_excel_bytes = output_converted_excel.getvalue()

            uploaded_file_name_base = uploaded_file.name.split('.')[0]
            st.download_button(
                label="Download Converted Report as Excel", # 下載轉換後的報告 (Excel)
                data=st.session_state.converted_excel_bytes,
                file_name=f"{uploaded_file_name_base}_converted.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            if 'Band' in df_converted_temp.columns:
                unique_bands_in_file = df_converted_temp['Band'].astype(str).str.strip().unique().tolist()
                if set(unique_bands_in_file) != set(st.session_state.get('unique_bands_in_file', [])):
                    st.session_state.unique_bands_in_file = unique_bands_in_file
                    st.rerun() 
        else:
            st.error("Conversion failed or no data could be extracted.") # 轉換失敗或無法提取資料。
            conversion_status_text.text("Conversion failed.") # 轉換失敗。
    else: 
        st.success("Report already converted.") # 報告已轉換。
        st.write("Converted Data Preview (first 5 rows):") # 轉換後資料預覽 (前 5 列)：
        st.dataframe(st.session_state.df_converted.head())
        uploaded_file_name_base = uploaded_file.name.split('.')[0]
        if st.session_state.converted_excel_bytes is not None:
            st.download_button(
                label="Download Converted Report as Excel", # 下載轉換後的報告 (Excel)
                data=st.session_state.converted_excel_bytes,
                file_name=f"{uploaded_file_name_base}_converted.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


st.header("2. Analyze Report") # 2. 分析報告

if st.session_state.df_converted is not None:
    if st.button("Start Analysis"): # 開始分析
        st.info("Starting analysis, please wait...") # 開始分析，請稍候...
        
        analysis_progress_bar = st.progress(0)
        analysis_status_text = st.empty()

        def update_analysis_progress(value):
            analysis_progress_bar.progress(value / 100)
            analysis_status_text.text(f"Analysis Progress: {value}%") # 分析進度

        current_band_target_powers_for_analysis = {
            band: st.session_state.band_target_powers_state.get(band, 23.0) 
            for band in st.session_state.band_target_powers_state
        }

        attention_required, found_categories_in_report, total_test_items_checked, original_columns_from_df, bands_checked = analyze_lte_report(
            st.session_state.df_converted, 
            current_band_target_powers_for_analysis, 
            st.session_state.power_tolerance, st.session_state.evm_threshold, 
            st.session_state.sem_threshold, st.session_state.aclr_threshold, 
            st.session_state.mpr_tolerance, update_analysis_progress
        )
        
        st.success("Analysis Complete!") # 分析完成！
        analysis_status_text.text("Analysis Complete.") # 分析完成。
        st.session_state.analysis_results = {
            'attention_required': attention_required,
            'found_categories_in_report': found_categories_in_report,
            'total_test_items_checked': total_test_items_checked,
            'original_columns_from_df': original_columns_from_df,
            'bands_checked': bands_checked 
        }
        st.session_state.analysis_done = True
        st.rerun() 


if st.session_state.analysis_done and st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results
    attention_required = results['attention_required']
    found_categories_in_report = results['found_categories_in_report']
    total_test_items_checked = results['total_test_items_checked']
    original_columns_from_df = results['original_columns_from_df']
    bands_checked = results['bands_checked'] 

    st.subheader("--- Overall Analysis Summary ---") # --- 整體分析摘要 ---
    st.write(f"Total test items checked: **{total_test_items_checked}**") # 總共檢查的測試項目：
    
    if bands_checked:
        st.write(f"Total bands checked: **{', '.join(bands_checked)}**") # 總共檢查的 Band：
    else:
        st.write("No specific bands found in the report or 'Band' column is missing.") # 報告中未找到特定 Band 或缺少 'Band' 欄位。

    total_attention_items_overall = sum(len(items) for items in attention_required.values())

    if total_attention_items_overall > 0:
        st.error(f"A total of **{total_attention_items_overall}** test items require extra attention.") # 總共有 **{total_attention_items_overall}** 個測試項目需要特別注意。
    else:
        st.success("All specified test items meet the standards. No items require attention.") # 所有指定的測試項目均符合標準。沒有項目需要注意。

    expected_categories = [ 
        "6.2.2 Maximum Output Power",
        "6.2.3 Maximum Power Reduction",
        "6.5.2.1 Error Vector Magnitude (EVM) for PUSCH",
        "6.6.2.1 Spectrum Emission Mask",
        "6.6.2.3 Adjacent Channel Leakage Power Ratio"
    ]
    st.markdown("### Category-wise Summary:") # ### 類別摘要：
    for category_key_in_summary in expected_categories:
        display_category_name_for_summary = category_key_in_summary
        if category_key_in_summary == "6.6.2.3 Adjacent Channel Leakage Power Ratio":
            display_category_name_for_summary = "6.6.2.3 Adjacent Channel Leakage Power Ratio (ACLR)"

        if category_key_in_summary not in found_categories_in_report:
             st.warning(f"- **{display_category_name_for_summary}**: Not Found in Report") # **{display_category_name_for_summary}**: 報告中未找到
        else:
            items_in_category = attention_required.get(category_key_in_summary, [])
            count = len(items_in_category)
            if count > 0:
                st.error(f"- **{display_category_name_for_summary}**: {count} items needing attention") # **{display_category_name_for_summary}**: {count} 個項目需要注意
            else:
                st.success(f"- **{display_category_name_for_summary}**: All good") # **{display_category_name_for_summary}**: 全部正常

    st.subheader("Detailed Attention Items") # 詳細注意項目
    
    # --- 修改 Detailed Table 的欄位名稱 ---
    dynamic_table_columns = list(original_columns_from_df) # 複製原始欄位列表

    # 移除不必要的舊欄位，並確保 'Measured', 'Criteria', 'Margin', '3GPP Status', 'Attention_Message' 的順序
    if 'Measured' not in dynamic_table_columns:
        # 如果原始資料沒有 Measured，則在尾部添加這些欄位
        dynamic_table_columns.extend(['Measured', 'Criteria', 'Margin', '3GPP Status', 'Attention_Message'])
        # 如果 Status 存在於原始欄位中，則移除，因為我們用 3GPP Status 取代
        if 'Status' in dynamic_table_columns:
            dynamic_table_columns.remove('Status')

    else: # 如果 'Measured' 存在
        measured_idx = dynamic_table_columns.index('Measured')
        
        # 移除舊的 Limit/Target/Threshold 欄位
        for col_to_remove in ['Limit Low', 'Limit High', 'Target Range', 'Threshold']:
            if col_to_remove in dynamic_table_columns:
                dynamic_table_columns.remove(col_to_remove)
        
        # 重新獲取 measured_idx，因為列表可能已改變
        if 'Measured' in dynamic_table_columns:
            measured_idx = dynamic_table_columns.index('Measured')
        else: # 理論上不應該發生，但作為防禦性程式碼
            measured_idx = -1 


        # 插入 'Criteria' 和 'Margin'
        if 'Margin' not in dynamic_table_columns:
            dynamic_table_columns.insert(measured_idx + 1 if measured_idx != -1 else len(dynamic_table_columns), 'Margin')
        if 'Criteria' not in dynamic_table_columns:
            dynamic_table_columns.insert(measured_idx + 1 if measured_idx != -1 else len(dynamic_table_columns), 'Criteria')

        # 確保 '3GPP Status' 存在且位置正確
        # 先移除舊的 'Status' (如果存在)
        if 'Status' in dynamic_table_columns:
            dynamic_table_columns.remove('Status')
        
        # 添加 '3GPP Status' (如果不存在) 並調整其位置
        if '3GPP Status' not in dynamic_table_columns:
            # 嘗試將其放在 Measured, Criteria, Margin 之後
            insert_pos = measured_idx + 3 if measured_idx != -1 and measured_idx + 3 <= len(dynamic_table_columns) else len(dynamic_table_columns)
            if 'Attention_Message' in dynamic_table_columns: # 如果 Attention_Message 已存在，則在其之前插入
                 try:
                     insert_pos = dynamic_table_columns.index('Attention_Message')
                 except ValueError:
                     pass # 保持在列表末尾（Attention_Message之前）
            dynamic_table_columns.insert(insert_pos, '3GPP Status')


    # 確保 'Attention_Message' 在最後
    if 'Attention_Message' in dynamic_table_columns:
        dynamic_table_columns.remove('Attention_Message')
    dynamic_table_columns.append('Attention_Message')

    # 確保欄位不重複
    dynamic_table_columns = list(dict.fromkeys(dynamic_table_columns))


    table_data = []
    for category_key, items in attention_required.items():
        for item_data in items: # item_data 包含原始 CSV 的鍵，例如 'Status'
            row_values_ordered = []
            for col_display_name in dynamic_table_columns: # col_display_name 可能是 '3GPP Status'
                # 如果顯示的欄位名稱是 '3GPP Status'，則從 item_data 中獲取 'Status' 的值
                data_key = 'Status' if col_display_name == '3GPP Status' else col_display_name
                value = item_data.get(data_key, '')
                row_values_ordered.append(value)
            table_data.append(row_values_ordered)

    if table_data:
        df_results = pd.DataFrame(table_data, columns=dynamic_table_columns)
        
        st.dataframe(df_results, use_container_width=True, height=400) 

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_results.to_excel(writer, index=False, sheet_name='Detailed Report')
        output.seek(0)
        
        st.download_button(
            label="Download Detailed Report as Excel", # 下載詳細報告 (Excel)
            data=output,
            file_name="lte_analysis_detailed_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No items requiring attention were found based on the current thresholds.") # 根據目前閾值，未找到需要注意的項目。

else:
    st.info("Please upload a raw CSV file first to start the analysis.") # 請先上傳原始 CSV 檔案以開始分析。

