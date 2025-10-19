# Transformer Usage Analysis

## Overview
This document maps where each custom transformer is used in the exoplanet detection pipeline.

---

## 📍 Transformer Locations and Usage

### 1. **GalacticCoordinatesTransformer**
**Location:** `common/transformers.py` (lines 9-46)

**Purpose:** Converts celestial coordinates (RA, Dec) to galactic coordinates using astropy, with sine/cosine encoding for angle wrapping.

**Where Used:**
- ✅ **Initialized in:** `common/base_pipeline.py` (line 27)
  ```python
  self.galactic_transformer = GalacticCoordinatesTransformer()
  ```
  
- ✅ **Applied in:** `common/base_pipeline.py` (line 116) - inside `_common_preprocessing()`
  ```python
  df = self.galactic_transformer.transform(df)
  ```

- ✅ **Exported from:** `common/__init__.py` (lines 4, 12)

**Status:** ✅ **ACTIVELY USED** - Applied to all models during common preprocessing

**What it does:**
- Converts `merged_ra` and `merged_dec` to galactic coordinates (`merged_gal_l`, `merged_gal_b`)
- Encodes angles as sine/cosine pairs for better ML handling
- Drops original RA/Dec columns (telescope-dependent)

---

### 2. **DepthToRadiusRatioTransformer**
**Location:** `common/transformers.py` (lines 48-80)

**Purpose:** Converts transit depth to radius ratio with error propagation.

**Where Used:**
- ✅ **Initialized in:** `common/base_pipeline.py` (line 25)
  ```python
  self.depth_to_radius_transformer = DepthToRadiusRatioTransformer()
  ```
  
- ✅ **Applied in:** `common/base_pipeline.py` (line 110) - inside `_common_preprocessing()`
  ```python
  df = self.depth_to_radius_transformer.transform(df)
  ```

**Status:** ✅ **ACTIVELY USED** - Applied to all models during common preprocessing

**What it does:**
- Converts `merged_koi_depth` → `merged_koi_ror` (radius ratio = sqrt(depth))
- Propagates errors: `merged_koi_depth_err1/2` → `merged_koi_ror_err1/2`
- Drops original depth columns after conversion

---

### 3. **CrossMissionDuplicateRemover**
**Location:** `common/transformers.py` (lines 82-176)

**Purpose:** Detects and removes duplicate exoplanets observed by different missions (Kepler vs TESS).

**Where Used:**
- ✅ **Initialized in:** `common/base_pipeline.py` (line 26)
  ```python
  self.duplicate_remover = CrossMissionDuplicateRemover()
  ```
  
- ✅ **Applied in:** `common/base_pipeline.py` (line 113) - inside `_common_preprocessing()`
  ```python
  df = self.duplicate_remover.transform(df)
  ```

**Status:** ✅ **ACTIVELY USED** - Applied to all models during common preprocessing

**What it does:**
- Matches duplicates using spatial (RA/Dec) and period tolerances
- Priority: CONFIRMED > FALSE POSITIVE > CANDIDATE
- Prefers TESS over Kepler when disposition is equal
- Default tolerances: 0.001° for RA/Dec (~3.6 arcsec), 0.01 days for period

---

### 4. **RadiusRatioTransformer**
**Location:** `common/transformers.py` (lines 178-198)

**Purpose:** Creates ratio features from planet and star radii.

**Where Used:**
- ⚠️ **Exported from:** `common/__init__.py` (lines 5, 13)
- ❌ **NOT initialized or applied anywhere in the codebase**

**Status:** ❌ **UNUSED** - Defined but never instantiated or called

**What it would do (if used):**
- Create `merged_radius_ratio` = `merged_koi_prad / merged_koi_srad`
- Apply log transformation: `np.log1p(radius_ratio)`

**Note:** Similar functionality exists in model-specific pipelines:
- CatBoost: Creates `radius_ratio_squared` if `radius_ratio_calculated` exists
- LightGBM: Creates `lgbm_prad_srad_interaction` directly

---

### 5. **TemperatureFeatureTransformer**
**Location:** `common/transformers.py` (lines 200-225)

**Purpose:** Creates temperature-related features for habitability analysis.

**Where Used:**
- ⚠️ **Exported from:** `common/__init__.py` (lines 6, 14)
- ❌ **NOT initialized or applied anywhere in the codebase**

**Status:** ❌ **UNUSED** - Defined but never instantiated or called

**What it would do (if used):**
- Create `merged_temp_ratio` = `merged_koi_teq / merged_koi_steff`
- Create `merged_temp_diff` = `merged_koi_steff - merged_koi_teq`
- Create `merged_log_teq` and `merged_log_steff`

**Note:** Similar functionality exists in model-specific pipelines:
- LightGBM: Creates `lgbm_teq_insol_interaction` and `lgbm_teq_div_insol`
- CatBoost: Creates `energy_balance` using insol/teq ratio

---

## 🔄 Pipeline Flow

### Common Preprocessing (All Models)
```python
BasePipeline._common_preprocessing() performs:
1. Filter to "merged_*" columns (+ keep 'mission' if present)
2. Encode labels (training) or drop disposition (inference)
3. Fix physically impossible values (negative → NaN)
4. Remove exact duplicates
5. ✅ DepthToRadiusRatioTransformer
6. ✅ CrossMissionDuplicateRemover
7. ✅ GalacticCoordinatesTransformer
```

### Model-Specific Preprocessing
Each model's pipeline class inherits from `BasePipeline` and implements:
- `CatBoostPipeline._model_specific_preprocessing()`
- `LightGBMPipeline._model_specific_preprocessing()`
- `RandomForestPipeline._model_specific_preprocessing()`

These add model-specific feature engineering (interactions, polynomials, etc.)

---

## 📊 Summary Table

| Transformer | Location | Initialized | Applied | Status |
|------------|----------|-------------|---------|--------|
| **GalacticCoordinatesTransformer** | transformers.py:9 | base_pipeline.py:27 | base_pipeline.py:116 | ✅ ACTIVE |
| **DepthToRadiusRatioTransformer** | transformers.py:48 | base_pipeline.py:25 | base_pipeline.py:110 | ✅ ACTIVE |
| **CrossMissionDuplicateRemover** | transformers.py:82 | base_pipeline.py:26 | base_pipeline.py:113 | ✅ ACTIVE |
| **RadiusRatioTransformer** | transformers.py:178 | ❌ None | ❌ None | ⚠️ UNUSED |
| **TemperatureFeatureTransformer** | transformers.py:200 | ❌ None | ❌ None | ⚠️ UNUSED |

---

## 💡 Recommendations

### Option 1: Integrate Unused Transformers
Add `RadiusRatioTransformer` and `TemperatureFeatureTransformer` to the common preprocessing pipeline:

```python
# In BasePipeline.__init__()
self.radius_ratio_transformer = RadiusRatioTransformer()
self.temperature_transformer = TemperatureFeatureTransformer()

# In BasePipeline._common_preprocessing()
df = self.radius_ratio_transformer.transform(df)
df = self.temperature_transformer.transform(df)
```

**Pros:**
- More standardized features across all models
- Reusable, well-tested transformations
- Physically meaningful features

**Cons:**
- Adds features that models may already create in their own way
- Could duplicate feature engineering

### Option 2: Remove Unused Transformers
Remove `RadiusRatioTransformer` and `TemperatureFeatureTransformer`:
- Delete from `common/transformers.py`
- Remove from `common/__init__.py` exports

**Pros:**
- Cleaner codebase
- No confusion about unused code

**Cons:**
- Lose potentially useful transformers
- Would need to rewrite if needed later

### Option 3: Keep As-Is (Documentation Only)
Document that these transformers are available but not currently integrated:
- Keep them in `transformers.py` for future use
- Update docstrings to indicate they're optional/available

**Pros:**
- Preserves flexibility
- No breaking changes

**Cons:**
- Dead code remains in codebase
- May confuse future developers

---

## 🔍 How to Trace Transformer Usage

### Find where a transformer is defined:
```bash
grep -n "class TransformerName" common/transformers.py
```

### Find where it's initialized:
```bash
grep -rn "TransformerName()" .
```

### Find where it's called:
```bash
grep -rn "\.transform(" common/base_pipeline.py
```

### Check imports:
```bash
grep -rn "from common.transformers import" .
grep -rn "from common import" .
```

---

*Generated: 2025-10-05*
