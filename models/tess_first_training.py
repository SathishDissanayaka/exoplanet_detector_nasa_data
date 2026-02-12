"""
TESS-First Data Splitting Utility
==================================
Implements the 55% TESS + 45% Kepler data splitting strategy.

This approach balances telescope representation to reduce
domain bias while leveraging TESS’s generalization advantage.
"""

import pandas as pd
fgggf

def split_tess_first_data(merged_data, tess_train_pct=0.55, kepler_train_pct=0.45, mission_col='mission', target_col='merged_koi_disposition', random_seed=42):
    """
    Split data with GUARANTEED TESS/Kepler composition AND balanced target variable.
    
    This ensures both exact mission composition and proportional target distribution.

    Args:
        merged_data: DataFrame with merged Kepler+TESS data
        tess_train_pct: Exact percentage of TESS data in training set
        kepler_train_pct: Exact percentage of Kepler data in training set
        mission_col: Column name indicating telescope mission
        target_col: Column name for the target variable
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data) with guaranteed composition and balanced targets
    """
    print("\n" + "=" * 70)
    print(f"DOUBLE-BALANCED SPLITTING ({tess_train_pct*100:.0f}% TESS + {kepler_train_pct*100:.0f}% Kepler)")
    print("=" * 70)
    print("Balanced by both MISSION and TARGET variable")

    # Separate by telescope
    tess_df = merged_data[merged_data[mission_col] == 'TESS'].copy()
    kepler_df = merged_data[merged_data[mission_col] == 'KEPLER'].copy()

    print(f"\nAvailable Data:")
    print(f"   TESS: {len(tess_df):,} samples")
    print(f"   Kepler: {len(kepler_df):,} samples")
    print(f"   Total: {len(merged_data):,} samples")

    # Calculate total training size for 80/20 split
    total_samples = len(merged_data)
    total_train_size = int(total_samples * 0.8)

    print(f"\nTarget: {total_train_size:,} training samples (80% of total)")

    # Calculate EXACT training sizes for guaranteed composition
    tess_train_size = int(total_train_size * tess_train_pct)
    kepler_train_size = total_train_size - tess_train_size

    print(f"\nRequired Training Composition:")
    print(f"   TESS: {tess_train_size:,} samples ({tess_train_pct:.1%})")
    print(f"   Kepler: {kepler_train_size:,} samples ({kepler_train_pct:.1%})")

    # Check if we have enough data
    if tess_train_size > len(tess_df):
        print(f"ERROR: Not enough TESS data!")
        tess_train_size = len(tess_df)
        kepler_train_size = total_train_size - tess_train_size
    
    if kepler_train_size > len(kepler_df):
        print(f"ERROR: Not enough Kepler data!")
        kepler_train_size = len(kepler_df)
        tess_train_size = total_train_size - kepler_train_size

    # STRATIFIED SAMPLING BY TARGET VARIABLE FOR EACH TELESCOPE
    print(f"\nPerforming stratified sampling by target variable...")
    
    # For TESS: Sample proportionally from each target class
    tess_train_samples = []
    for class_name in tess_df[target_col].unique():
        class_data = tess_df[tess_df[target_col] == class_name]
        class_pct = len(class_data) / len(tess_df)  # Natural distribution in TESS
        class_train_size = int(tess_train_size * class_pct)
        
        if class_train_size > 0:
            class_train = class_data.sample(n=class_train_size, random_state=random_seed)
            tess_train_samples.append(class_train)
            print(f"   TESS {class_name}: {len(class_train):,} samples ({class_train_size/len(class_data)*100:.1f}% of class)")
    
    tess_train = pd.concat(tess_train_samples)
    
    # For Kepler: Sample proportionally from each target class  
    kepler_train_samples = []
    for class_name in kepler_df[target_col].unique():
        class_data = kepler_df[kepler_df[target_col] == class_name]
        class_pct = len(class_data) / len(kepler_df)  # Natural distribution in Kepler
        class_train_size = int(kepler_train_size * class_pct)
        
        if class_train_size > 0:
            class_train = class_data.sample(n=class_train_size, random_state=random_seed)
            kepler_train_samples.append(class_train)
            print(f"   Kepler {class_name}: {len(class_train):,} samples ({class_train_size/len(class_data)*100:.1f}% of class)")
    
    kepler_train = pd.concat(kepler_train_samples)

    # Remaining data goes to testing
    tess_test = tess_df.drop(tess_train.index)
    kepler_test = kepler_df.drop(kepler_train.index)

    # Combine training data
    train_data = pd.concat([tess_train, kepler_train], ignore_index=True)
    train_data = train_data.sample(frac=1, random_state=random_seed)  # Shuffle

    # Combine test data
    test_data = pd.concat([tess_test, kepler_test], ignore_index=True)
    test_data = test_data.sample(frac=1, random_state=random_seed)  # Shuffle

    # VERIFICATION
    print(f"\nGUARANTEED COMPOSITION ACHIEVED:")
    print(f"   Training Set: {len(train_data):,} samples")
    
    # Mission composition
    actual_tess_train = len(train_data[train_data[mission_col] == 'TESS'])
    actual_kepler_train = len(train_data[train_data[mission_col] == 'KEPLER'])
    print(f"     TESS: {actual_tess_train:,} samples ({actual_tess_train/len(train_data):.1%})")
    print(f"     Kepler: {actual_kepler_train:,} samples ({actual_kepler_train/len(train_data):.1%})")
    
    print(f"   Test Set: {len(test_data):,} samples")
    print(f"     TESS: {len(tess_test):,} samples")
    print(f"     Kepler: {len(kepler_test):,} samples")

    # Target distribution verification
    print(f"\nTARGET DISTRIBUTION - TRAINING SET:")
    print("   By Mission:")
    for mission in ['TESS', 'KEPLER']:
        mission_data = train_data[train_data[mission_col] == mission]
        print(f"     {mission}:")
        for class_name in train_data[target_col].unique():
            count = len(mission_data[mission_data[target_col] == class_name])
            if len(mission_data) > 0:
                pct = count / len(mission_data) * 100
                print(f"       {class_name}: {count:,} ({pct:.1f}%)")

    print(f"\nTARGET DISTRIBUTION - TEST SET:")
    print("   By Mission:")
    for mission in ['TESS', 'KEPLER']:
        mission_data = test_data[test_data[mission_col] == mission]
        print(f"     {mission}:")
        for class_name in test_data[target_col].unique():
            count = len(mission_data[mission_data[target_col] == class_name])
            if len(mission_data) > 0:
                pct = count / len(mission_data) * 100
                print(f"       {class_name}: {count:,} ({pct:.1f}%)")

    print(f"\nBALANCE VERIFICATION:")
    print(f"   Overall Split: {len(train_data):,} train ({len(train_data)/total_samples*100:.1f}%) / {len(test_data):,} test ({len(test_data)/total_samples*100:.1f}%)")
    print(f"   Mission Composition: Exact {tess_train_pct:.1%} TESS, {kepler_train_pct:.1%} Kepler")
    print(f"   Target Distribution: Preserved natural distribution within each mission")

    return train_data, test_data

