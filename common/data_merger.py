"""
Data merger for combining Kepler and TESS telescope datasets.
This is the first step before any model-specific preprocessing.
"""
import pandas as pd

class DatasetMerger:
    """
    Handles integrating of Kepler and TESS telescope datasets.
    This runs BEFORE any model-specific preprocessing.
    """
    
    # Mapping from TESS column names to KOI column names
    TESS_TO_KOI_MAPPING = {
        "tfopwg_disp": "koi_disposition",
        "ra": "ra",
        "dec": "dec",
        "pl_tranmid": "koi_time0",
        "pl_tranmiderr1": "koi_time0_err1",
        "pl_tranmiderr2": "koi_time0_err2",
        "pl_orbper": "koi_period",
        "pl_orbpererr1": "koi_period_err1",
        "pl_orbpererr2": "koi_period_err2",
        "pl_trandurh": "koi_duration",
        "pl_trandurherr1": "koi_duration_err1",
        "pl_trandurherr2": "koi_duration_err2",
        "pl_trandep": "koi_depth",
        "pl_trandeperr1": "koi_depth_err1",
        "pl_trandeperr2": "koi_depth_err2",
        "pl_rade": "koi_prad",
        "pl_radeerr1": "koi_prad_err1",
        "pl_radeerr2": "koi_prad_err2",
        "pl_insol": "koi_insol",
        "pl_insolerr1": "koi_insol_err1",
        "pl_insolerr2": "koi_insol_err2",
        "pl_eqt": "koi_teq",
        "pl_eqterr1": "koi_teq_err1",
        "pl_eqterr2": "koi_teq_err2",
        "st_teff": "koi_steff",
        "st_tefferr1": "koi_steff_err1",
        "st_tefferr2": "koi_steff_err2",
        "st_logg": "koi_slogg",
        "st_loggerr1": "koi_slogg_err1",
        "st_loggerr2": "koi_slogg_err2",
        "st_rad": "koi_srad",
        "st_raderr1": "koi_srad_err1",
        "st_raderr2": "koi_srad_err2"
    }
    
    # Mapping TESS disposition codes to KOI-like labels
    TESS_DISPOSITION_MAPPING = {
        "FP": "FALSE POSITIVE",
        "FA": "FALSE POSITIVE",
        "PC": "CANDIDATE",
        "APC": "CANDIDATE",
        "CP": "CONFIRMED",
        "KP": "CONFIRMED"
    }
    
    def __init__(self):
        self.merge_stats = {}
    
    def merge(self, kepler_data: pd.DataFrame, tess_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Kepler and TESS datasets with intelligent column mapping.
        
        Args:
            kepler_data: Kepler telescope dataset (KOI)
            tess_data: TESS telescope dataset (TOI)
            
        Returns:
            Merged DataFrame with unified column names and mission tags
        """
        # Store initial stats
        self.merge_stats = {
            'kepler_rows': len(kepler_data),
            'tess_rows': len(tess_data),
            'kepler_cols': len(kepler_data.columns),
            'tess_cols': len(tess_data.columns)
        }
        
        # Work with copies to avoid modifying original data
        koi = kepler_data.copy()
        toi = tess_data.copy()
        
        # Step 1: Preserve raw disposition columns
        if 'koi_disposition' in koi.columns:
            koi['koi_raw_disposition'] = koi['koi_disposition']
        
        if 'tfopwg_disp' in toi.columns:
            toi['tess_raw_disposition'] = toi['tfopwg_disp']
        
        # Step 2: Add mission tags
        koi['mission'] = 'KEPLER'
        toi['mission'] = 'TESS'
        
        # Step 3: Create unified "merged_" columns based on mapping
        created_cols = []
        
        for tess_col, koi_col in self.TESS_TO_KOI_MAPPING.items():
            unified_col = f"merged_{koi_col}"
            
            # Copy KOI values to unified column (if column exists)
            if koi_col in koi.columns:
                koi[unified_col] = koi[koi_col]
                created_cols.append(unified_col)
            
            # Copy TESS values to unified column (if column exists)
            if tess_col in toi.columns:
                toi[unified_col] = toi[tess_col]
                if unified_col not in created_cols:
                    created_cols.append(unified_col)
        
        # Store what was created
        self.merge_stats['columns_created'] = created_cols
        
        # Step 4: Special handling for disposition (use mapped version for TESS)
        if 'tfopwg_disp' in toi.columns:
            toi['merged_koi_disposition'] = toi['tfopwg_disp'].map(self.TESS_DISPOSITION_MAPPING)
        if 'koi_disposition' in koi.columns:
            koi['merged_koi_disposition'] = koi['koi_disposition']
        
        # Step 5: Concatenate datasets
        merged_df = pd.concat([koi, toi], ignore_index=True)
        
        # Update final stats
        self.merge_stats['merged_rows'] = len(merged_df)
        self.merge_stats['merged_cols'] = len(merged_df.columns)
        self.merge_stats['kepler_rows_after_merge'] = len(merged_df[merged_df['mission'] == 'KEPLER'])
        self.merge_stats['tess_rows_after_merge'] = len(merged_df[merged_df['mission'] == 'TESS'])
        
        # Store info about unified columns created
        merged_cols = [col for col in merged_df.columns if col.startswith('merged_')]
        self.merge_stats['unified_columns_created'] = len(merged_cols)
        
        return merged_df
    
    def get_merge_info(self) -> dict:
        """Get detailed information about the last merge operation"""
        return self.merge_stats
    
    def get_column_mapping(self) -> dict:
        """Get the TESS to KOI column mapping used"""
        return self.TESS_TO_KOI_MAPPING.copy()
    
    def get_disposition_mapping(self) -> dict:
        """Get the TESS disposition code mapping used"""
        return self.TESS_DISPOSITION_MAPPING.copy()


# Convenience function for quick merging
def merge_kepler_tess(kepler_data: pd.DataFrame, tess_data: pd.DataFrame) -> pd.DataFrame:
    """
    Quick function to merge Kepler and TESS data.
    
    Args:
        kepler_data: Kepler telescope dataset (KOI)
        tess_data: TESS telescope dataset (TOI)
        
    Returns:
        Merged DataFrame with unified columns and mission tags
    """
    merger = DatasetMerger()
    return merger.merge(kepler_data, tess_data)
