"""Custom transformers for feature engineering"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.preprocessing import StandardScaler, QuantileTransformer

class SafeMultiplicityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        if 'merged_multiplicity' in X.columns:
            # Create telescope-agnostic multiplicity categories
            X['feat_is_single_system'] = (X['merged_multiplicity'] == 1).astype(int)
            X['feat_is_multi_planet_system'] = (X['merged_multiplicity'] > 1).astype(int)
            X['feat_is_high_multiplicity'] = (X['merged_multiplicity'] > 2).astype(int)

            # Log transform for systems with many planets
            X['feat_log_multiplicity'] = np.log1p(X['merged_multiplicity'])
        return X

class ExoplanetPhysicsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    
    def transform(self, X):
        X = X.copy()

        # Temperature ratio (important for classification)
        if all(col in X.columns for col in ["merged_koi_teq", "merged_koi_steff"]):
            X["feat_temp_ratio"] = X["merged_koi_teq"] / X["merged_koi_steff"]

        # Planet equilibrium temperature theoretical vs measured
        if all(col in X.columns for col in ["merged_koi_teq", "merged_koi_insol"]):
            # Simple relationship: Teq ∝ insol^0.25
            X["feat_teq_theoretical"] = 280 * (X["merged_koi_insol"] ** 0.25)
            X["feat_teq_discrepancy"] = X["merged_koi_teq"] - X["feat_teq_theoretical"]

        return X

class SafeOrbitalFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        if "merged_koi_period" in X.columns:
            # Always create the feature, use NaN for invalid values
            X["feat_log_period"] = np.nan
            period_mask = X["merged_koi_period"] > 0
            X.loc[period_mask, "feat_log_period"] = np.log10(X.loc[period_mask, "merged_koi_period"])

            # Physical categories (always create)
            X["feat_is_ultra_short_period"] = (X["merged_koi_period"] < 1.0).astype(int)
            X["feat_is_habitable_zone_period"] = (
                (X["merged_koi_period"] > 50) & (X["merged_koi_period"] < 400)
            ).astype(int)
        return X

class SafeTransitDurationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        if all(col in X.columns for col in ["merged_koi_duration", "merged_koi_period"]):
            X["feat_duration_ratio"] = np.nan
            mask = (X["merged_koi_period"] > 0) & (X["merged_koi_duration"] > 0)
            X.loc[mask, "feat_duration_ratio"] = (
                X.loc[mask, "merged_koi_duration"] / (24 * X.loc[mask, "merged_koi_period"])
            )
        return X

class SafeTransitDepthTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        if "merged_koi_depth" in X.columns:
            X["feat_log_depth"] = np.nan
            depth_mask = X["merged_koi_depth"] > 0
            X.loc[depth_mask, "feat_log_depth"] = np.log10(X.loc[depth_mask, "merged_koi_depth"])

        # Depth SNR
        if all(col in X.columns for col in ["merged_koi_depth", "merged_koi_depth_err1", "merged_koi_depth_err2"]):
            X["feat_depth_snr"] = np.nan
            avg_err = (X["merged_koi_depth_err1"].abs() + X["merged_koi_depth_err2"].abs()) / 2
            snr_mask = (avg_err > 0) & (X["merged_koi_depth"] > 0)
            X.loc[snr_mask, "feat_depth_snr"] = X.loc[snr_mask, "merged_koi_depth"] / avg_err[snr_mask]
        return X

class SafePlanetRadiusTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        if all(col in X.columns for col in ["merged_koi_prad", "merged_koi_srad"]):
            X["feat_radius_ratio"] = np.nan
            mask = (X["merged_koi_srad"] > 0) & (X["merged_koi_prad"] > 0)
            X.loc[mask, "feat_radius_ratio"] = X.loc[mask, "merged_koi_prad"] / X.loc[mask, "merged_koi_srad"]

        if "merged_koi_prad" in X.columns:
            X["feat_log_prad"] = np.nan
            prad_mask = X["merged_koi_prad"] > 0
            X.loc[prad_mask, "feat_log_prad"] = np.log10(X.loc[prad_mask, "merged_koi_prad"])
        return X

class SafeEnvironmentTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        if "merged_koi_insol" in X.columns:
            X["feat_log_insol"] = np.nan
            insol_mask = X["merged_koi_insol"] > 0
            X.loc[insol_mask, "feat_log_insol"] = np.log10(X.loc[insol_mask, "merged_koi_insol"])

            X["feat_is_habitable_zone"] = (
                (X["merged_koi_insol"] > 0.32) & (X["merged_koi_insol"] < 1.78)
            ).astype(int)

        if "merged_koi_teq" in X.columns:
            X["feat_log_teq"] = np.nan
            teq_mask = X["merged_koi_teq"] > 0
            X.loc[teq_mask, "feat_log_teq"] = np.log10(X.loc[teq_mask, "merged_koi_teq"])
        return X

class SafeRelativeErrorTransformer(BaseEstimator, TransformerMixin):
    BASE_FEATURES = ["prad", "period", "depth", "duration", "teq", "insol", "steff", "slogg", "srad"]

    def fit(self, X, y=None):
        self.existing_error_pairs_ = []
        for base in self.BASE_FEATURES:
            col = f"merged_koi_{base}"
            e1, e2 = f"{col}_err1", f"{col}_err2"
            if col in X.columns and e1 in X.columns and e2 in X.columns:
                self.existing_error_pairs_.append((col, e1, e2))
        return self

    def transform(self, X):
        X = X.copy()
        for col, e1, e2 in self.existing_error_pairs_:
            new_col = f"feat_{col.split('_')[-1]}_rel_uncertainty"
            X[new_col] = np.nan
            mask = (X[col].notna()) & (X[col] != 0) & (X[e1].notna()) & (X[e2].notna())
            if mask.any():
                avg_err = (X.loc[mask, e1].abs() + X.loc[mask, e2].abs()) / 2
                relative_uncertainty = avg_err / X.loc[mask, col].abs()

                # Log-transform relative uncertainties to compress dynamic range
                X.loc[mask, new_col] = np.log10(relative_uncertainty + 1e-6)
        return X

class SignalToNoiseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()

        # Transit depth SNR
        if all(col in X.columns for col in ["merged_koi_depth", "merged_koi_depth_err1"]):
            X["feat_depth_snr"] = np.nan
            snr_mask = (X["merged_koi_depth_err1"] > 0) & (X["merged_koi_depth"] > 0)
            X.loc[snr_mask, "feat_depth_snr"] = (
                X.loc[snr_mask, "merged_koi_depth"] / X.loc[snr_mask, "merged_koi_depth_err1"]
            )

        # Transit duration SNR
        if all(col in X.columns for col in ["merged_koi_duration", "merged_koi_duration_err1"]):
            X["feat_duration_snr"] = np.nan
            snr_mask = (X["merged_koi_duration_err1"] > 0) & (X["merged_koi_duration"] > 0)
            X.loc[snr_mask, "feat_duration_snr"] = (
                X.loc[snr_mask, "merged_koi_duration"] / X.loc[snr_mask, "merged_koi_duration_err1"]
            )

        # Orbital period SNR
        if all(col in X.columns for col in ["merged_koi_period", "merged_koi_period_err1"]):
            X["feat_period_snr"] = np.nan
            snr_mask = (X["merged_koi_period_err1"] > 0) & (X["merged_koi_period"] > 0)
            X.loc[snr_mask, "feat_period_snr"] = (
                X.loc[snr_mask, "merged_koi_period"] / X.loc[snr_mask, "merged_koi_period_err1"]
            )

        # Planet radius SNR
        if all(col in X.columns for col in ["merged_koi_prad", "merged_koi_prad_err1"]):
            X["feat_prad_snr"] = np.nan
            snr_mask = (X["merged_koi_prad_err1"] > 0) & (X["merged_koi_prad"] > 0)
            X.loc[snr_mask, "feat_prad_snr"] = (
                X.loc[snr_mask, "merged_koi_prad"] / X.loc[snr_mask, "merged_koi_prad_err1"]
            )

        return X

class TelescopeAgnosticCleaner(BaseEstimator, TransformerMixin):
    """Remove ONLY truly telescope-leaking columns"""

    def __init__(self, drop_leaking=True):
        self.drop_leaking = drop_leaking
        # 🎯 Initialize telescope-leaking columns in __init__
        self.telescope_leaking_columns_ = [
            # Absolute timing - strongly telescope-dependent
            'merged_koi_time0', 'merged_koi_time0_err1', 'merged_koi_time0_err2',

            # Coordinates - observation location
            'merged_ra', 'merged_dec',

        ]

    def fit(self, X, y=None):
        # No need to redefine the list here since it's already in __init__
        return self

    def transform(self, X):
        X = X.copy()
        if self.drop_leaking:
            # Only drop truly telescope-leaking columns
            columns_to_drop = [col for col in self.telescope_leaking_columns_
                             if col in X.columns]
            X = X.drop(columns=columns_to_drop, errors="ignore")

            print(f"🔧 Removed {len(columns_to_drop)} telescope-leaking columns")
            if columns_to_drop:
                print(f"   Removed: {columns_to_drop}")
        return X


class RobustStellarScaler(BaseEstimator, TransformerMixin):
    """SafeStellarScaler that handles missing columns gracefully"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_ = []
        self.fitted_columns_ = []

    def fit(self, X, y=None):
        # Only use features that exist in the data
        stellar_features = ["merged_koi_steff", "merged_koi_slogg", "merged_koi_srad"]
        self.feature_names_ = [f for f in stellar_features if f in X.columns]

        if self.feature_names_:
            # Store the actual columns we're fitting on
            self.fitted_columns_ = self.feature_names_.copy()
            self.scaler.fit(X[self.feature_names_])
            print(f"  ✓ Fitted scaler on {len(self.feature_names_)} stellar features: {self.feature_names_}")
        else:
            print("  ⚠️  No stellar features found for scaling")
            self.fitted_columns_ = []

        return self

    def transform(self, X):
        X = X.copy()
        if self.fitted_columns_:
            # Only transform if we have the columns we fitted on
            available_columns = [col for col in self.fitted_columns_ if col in X.columns]
            missing_columns = set(self.fitted_columns_) - set(available_columns)

            if missing_columns:
                print(f"  ⚠️  Missing columns during transform: {missing_columns}")

            if available_columns:
                scaled_values = self.scaler.transform(X[available_columns])
                for i, col in enumerate(available_columns):
                    X[f"feat_norm_{col.split('_')[-1]}"] = scaled_values[:, i]

        return X

class RobustRelativeErrorTransformer(BaseEstimator, TransformerMixin):
    """SafeRelativeErrorTransformer that handles missing columns"""

    BASE_FEATURES = ["prad", "period", "depth", "duration", "teq", "insol", "steff", "slogg", "srad"]

    def fit(self, X, y=None):
        self.existing_error_pairs_ = []
        for base in self.BASE_FEATURES:
            col = f"merged_koi_{base}"
            e1, e2 = f"{col}_err1", f"{col}_err2"
            if col in X.columns and e1 in X.columns and e2 in X.columns:
                self.existing_error_pairs_.append((col, e1, e2))

        print(f"  ✓ Found {len(self.existing_error_pairs_)} features for relative error calculation")
        return self

    def transform(self, X):
        X = X.copy()
        for col, e1, e2 in self.existing_error_pairs_:
            # Only process if columns exist in this dataset
            if col in X.columns and e1 in X.columns and e2 in X.columns:
                new_col = f"feat_{col.split('_')[-1]}_rel_uncertainty"
                X[new_col] = np.nan
                mask = (X[col].notna()) & (X[col] != 0) & (X[e1].notna()) & (X[e2].notna())
                if mask.any():
                    avg_err = (X.loc[mask, e1].abs() + X.loc[mask, e2].abs()) / 2
                    relative_uncertainty = avg_err / X.loc[mask, col].abs()
                    X.loc[mask, new_col] = np.log10(relative_uncertainty + 1e-6)

        return X


class TelescopeDistributionMatcher(BaseEstimator, TransformerMixin):
    """
    Advanced domain adaptation: Quantile normalization per telescope to align distributions
    """

    def __init__(self, features_to_normalize=None):
        self.features_to_normalize = features_to_normalize or [
            'merged_koi_insol', 'merged_koi_prad', 'merged_koi_srad',
            'merged_koi_steff', 'merged_koi_period', 'merged_koi_teq'
        ]
        self.quantile_transformers_ = {}
        self.telescope_col_ = 'mission'

    def fit(self, X, y=None):
        self.quantile_transformers_ = {}

        for feat in self.features_to_normalize:
            if feat in X.columns:
                self.quantile_transformers_[feat] = {}
                for telescope in X[self.telescope_col_].unique():
                    mask = X[self.telescope_col_] == telescope
                    if mask.sum() > 0:
                        qt = QuantileTransformer(output_distribution='normal', random_state=42)
                        qt.fit(X.loc[mask, [feat]])
                        self.quantile_transformers_[feat][telescope] = qt

        print(f"  ✓ Fitted quantile normalization for {len(self.quantile_transformers_)} features")
        return self

    def transform(self, X):
        X = X.copy()

        for feat, telescope_dict in self.quantile_transformers_.items():
            if feat in X.columns:
                normalized_feat = np.zeros(len(X))
                for telescope, qt in telescope_dict.items():
                    mask = X[self.telescope_col_] == telescope
                    if mask.sum() > 0:
                        normalized_feat[mask] = qt.transform(X.loc[mask, [feat]]).flatten()
                X[f"feat_qn_{feat}"] = normalized_feat

        return X

class RobustUncertaintyCompressor(BaseEstimator, TransformerMixin):
    """
    Compress uncertainty features to reduce telescope-specific precision differences
    """

    def __init__(self, compression_method='log_bin'):
        self.compression_method = compression_method
        self.uncertainty_features_ = []

    def fit(self, X, y=None):
        # Find all relative uncertainty features
        self.uncertainty_features_ = [col for col in X.columns if 'rel_uncertainty' in col]
        print(f"  ✓ Found {len(self.uncertainty_features_)} uncertainty features for compression")
        return self

    def transform(self, X):
        X = X.copy()

        for feat in self.uncertainty_features_:
            if feat in X.columns:
                if self.compression_method == 'log_bin':
                    # Option A: Log transform + binning
                    log_uncertainty = np.sign(X[feat]) * np.log1p(np.abs(X[feat].fillna(0)))
                    # Bin into 5 categories
                    try:
                        X[f"feat_compress_{feat}"] = pd.qcut(log_uncertainty, q=5, duplicates='drop', labels=False)
                    except:
                        X[f"feat_compress_{feat}"] = 0

                elif self.compression_method == 'robust_scale':
                    # Option B: Robust scaling with median
                    median_val = X[feat].median()
                    mad_val = (X[feat] - median_val).abs().median()
                    if mad_val > 0:
                        X[f"feat_compress_{feat}"] = (X[feat] - median_val) / mad_val
                    else:
                        X[f"feat_compress_{feat}"] = 0

        return X

class DomainInvariantProjector(BaseEstimator, TransformerMixin):
    """
    Create domain-invariant features using PCA on telescope-aligned features
    """

    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pca_ = None
        self.domain_features_ = []

    def fit(self, X, y=None):
        # Use features that are already telescope-normalized
        domain_features = [col for col in X.columns if any(x in col for x in ['feat_qn_', 'feat_compress_', 'feat_norm_'])]
        self.domain_features_ = [f for f in domain_features if f in X.columns]

        if len(self.domain_features_) >= self.n_components:
            self.pca_ = PCA(n_components=self.n_components, random_state=42)
            self.pca_.fit(X[self.domain_features_])
            print(f"  ✓ Fitted PCA on {len(self.domain_features_)} domain-invariant features")
        else:
            print(f"  ⚠️  Not enough features for PCA ({len(self.domain_features_)} < {self.n_components})")

        return self

    def transform(self, X):
        X = X.copy()

        if self.pca_ is not None and len(self.domain_features_) >= self.n_components:
            available_features = [f for f in self.domain_features_ if f in X.columns]
            if len(available_features) >= self.n_components:
                pca_features = self.pca_.transform(X[available_features])
                for i in range(self.n_components):
                    X[f'feat_domain_pc_{i+1}'] = pca_features[:, i]

        return X

class CrossMissionDuplicateRemover(BaseEstimator, TransformerMixin):
    """
    Detects and removes duplicate exoplanets observed by different missions.
    Uses spatial (RA, Dec) and orbital period matching with configurable tolerances.
    Priority: CONFIRMED > FALSE POSITIVE > CANDIDATE, with Kepler preferred over TESS.
    """
    
    def __init__(self, ra_tol=0.001, dec_tol=0.001, period_tol=0.01):
        """
        Args:
            ra_tol: Right ascension tolerance in degrees (default: 0.001 ≈ 3.6 arcsec)
            dec_tol: Declination tolerance in degrees (default: 0.001)
            period_tol: Period tolerance in days (default: 0.01)
        """
        self.ra_tol = ra_tol
        self.dec_tol = dec_tol
        self.period_tol = period_tol
        self.drop_indices_ = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Check required columns
        required = ['merged_ra', 'merged_dec', 'merged_koi_period', 'mission', 'merged_koi_disposition']
        if not all(col in df.columns for col in required):
            return df
        
        # Drop rows missing critical info
        df = df.dropna(subset=['merged_ra', 'merged_dec', 'merged_koi_period'])
        
        # Sort for sequential comparison
        df = df.sort_values(by=['merged_ra', 'merged_dec', 'merged_koi_period']).reset_index(drop=True)
        
        # Find potential duplicates
        possible_dupes = []
        for i in range(len(df) - 1):
            ra_diff = abs(df.loc[i, 'merged_ra'] - df.loc[i+1, 'merged_ra'])
            dec_diff = abs(df.loc[i, 'merged_dec'] - df.loc[i+1, 'merged_dec'])
            period_diff = abs(df.loc[i, 'merged_koi_period'] - df.loc[i+1, 'merged_koi_period'])
            
            if (ra_diff < self.ra_tol and dec_diff < self.dec_tol and 
                period_diff < self.period_tol and 
                df.loc[i, 'mission'] != df.loc[i+1, 'mission']):
                
                possible_dupes.append({
                    'index_1': i,
                    'index_2': i+1,
                    'mission_1': df.loc[i, 'mission'],
                    'mission_2': df.loc[i+1, 'mission'],
                    'disposition_1': df.loc[i, 'merged_koi_disposition'],
                    'disposition_2': df.loc[i+1, 'merged_koi_disposition']
                })
        
        if len(possible_dupes) == 0:
            return df
        
        # Pick the best entry for each duplicate pair
        dupe_df = pd.DataFrame(possible_dupes)
        
        def pick_best(row):
            # Priority: CONFIRMED > FALSE POSITIVE > CANDIDATE
            priority = {'CONFIRMED': 3, 'FALSE POSITIVE': 2, 'CANDIDATE': 1}
            p1 = priority.get(row['disposition_1'], 0)
            p2 = priority.get(row['disposition_2'], 0)
            
            if p1 > p2:
                return row['index_1']
            elif p2 > p1:
                return row['index_2']
            else:
                # Equal priority - prefer TESS over Kepler
                if row['mission_1'] == 'TESS':
                    return row['index_1']
                elif row['mission_2'] == 'TESS':
                    return row['index_2']
                else:
                    return row['index_1']
        
        dupe_df['keep_index'] = dupe_df.apply(pick_best, axis=1)
        dupe_df['drop_index'] = dupe_df.apply(
            lambda row: row['index_2'] if row['keep_index'] == row['index_1'] else row['index_1'],
            axis=1
        )
        
        # Drop duplicates
        drop_indices = dupe_df['drop_index'].tolist()
        self.drop_indices_ = drop_indices
        df_cleaned = df.drop(index=drop_indices).reset_index(drop=True)
        
        return df_cleaned

