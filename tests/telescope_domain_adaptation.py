"""
Telescope Domain Adaptation Analysis
=====================================
This script performs comprehensive analysis of cross-telescope transfer learning:
1. Feature distribution analysis between Kepler and TESS
2. Domain adaptation experiments (80/20 mixed training)
3. Reverse transfer (TESS→Kepler and Kepler→TESS)
4. Telescope-specific baseline models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import our project modules
from common.data_merger import DatasetMerger


class TelescopeDomainAdapter:
    """Handles domain adaptation experiments between Kepler and TESS"""
    
    def __init__(self, kepler_data, tess_data):
        self.kepler_data = kepler_data.copy()
        self.tess_data = tess_data.copy()
        self.merger = DatasetMerger()
        self.results = {}
        
        # Feature columns to analyze (merged columns only)
        self.feature_cols = None
        self.target_col = 'merged_koi_disposition'
        
    def prepare_data(self):
        """Merge and prepare datasets"""
        print("=" * 70)
        print("PREPARING DATA")
        print("=" * 70)
        
        # Merge datasets
        merged = self.merger.merge(self.kepler_data, self.tess_data)
        print(f"\n✅ Merged dataset shape: {merged.shape}")
        
        # Identify feature columns (all merged_* except disposition)
        self.feature_cols = [col for col in merged.columns 
                            if col.startswith('merged_') 
                            and col != self.target_col
                            and merged[col].dtype in ['float64', 'int64']]
        
        print(f"✅ Found {len(self.feature_cols)} feature columns")
        
        # Split by telescope
        self.kepler_df = merged[merged['mission'] == 'KEPLER'].copy()
        self.tess_df = merged[merged['mission'] == 'TESS'].copy()
        
        print(f"\n📊 Data Split:")
        print(f"  Kepler samples: {len(self.kepler_df)}")
        print(f"  TESS samples: {len(self.tess_df)}")
        
        # Clean data
        self._clean_datasets()
        
        return merged
    
    def _clean_datasets(self):
        """Remove rows with missing target or too many missing features"""
        for name, df in [('Kepler', self.kepler_df), ('TESS', self.tess_df)]:
            initial_rows = len(df)
            
            # Remove rows with missing target
            df_clean = df.dropna(subset=[self.target_col])
            
            # Remove rows with >50% missing features
            missing_threshold = len(self.feature_cols) * 0.5
            df_clean = df_clean[df_clean[self.feature_cols].notna().sum(axis=1) >= missing_threshold]
            
            if name == 'Kepler':
                self.kepler_df = df_clean
            else:
                self.tess_df = df_clean
                
            print(f"  {name}: {initial_rows} → {len(df_clean)} rows after cleaning")
    
    def analyze_feature_distributions(self):
        """Analyze and visualize feature distributions between telescopes"""
        print("\n" + "=" * 70)
        print("FEATURE DISTRIBUTION ANALYSIS")
        print("=" * 70)
        
        results = []
        
        for feature in self.feature_cols:
            kepler_vals = self.kepler_df[feature].dropna()
            tess_vals = self.tess_df[feature].dropna()
            
            if len(kepler_vals) < 10 or len(tess_vals) < 10:
                continue
                
            # Statistical tests
            ks_stat, ks_pval = stats.ks_2samp(kepler_vals, tess_vals)
            
            # Distribution metrics
            kepler_mean = kepler_vals.mean()
            tess_mean = tess_vals.mean()
            kepler_std = kepler_vals.std()
            tess_std = tess_vals.std()
            
            # Calculate distribution shift
            mean_diff_pct = abs(kepler_mean - tess_mean) / (abs(kepler_mean) + 1e-10) * 100
            
            results.append({
                'feature': feature,
                'kepler_mean': kepler_mean,
                'kepler_std': kepler_std,
                'tess_mean': tess_mean,
                'tess_std': tess_std,
                'mean_diff_pct': mean_diff_pct,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'significant_diff': ks_pval < 0.05
            })
        
        self.dist_analysis = pd.DataFrame(results).sort_values('ks_statistic', ascending=False)
        
        # Display top differences
        print("\n🔍 Features with Largest Distribution Differences:")
        print(self.dist_analysis[['feature', 'ks_statistic', 'ks_pvalue', 'mean_diff_pct']].head(10).to_string(index=False))
        
        # Summary statistics
        n_significant = (self.dist_analysis['ks_pvalue'] < 0.05).sum()
        print(f"\n📊 Summary:")
        print(f"  Total features analyzed: {len(self.dist_analysis)}")
        print(f"  Significantly different (p<0.05): {n_significant} ({n_significant/len(self.dist_analysis)*100:.1f}%)")
        
        self.results['distribution_analysis'] = self.dist_analysis
        
        return self.dist_analysis
    
    def train_baseline_models(self):
        """Train telescope-specific baseline models"""
        print("\n" + "=" * 70)
        print("TRAINING TELESCOPE-SPECIFIC BASELINES")
        print("=" * 70)
        
        baselines = {}
        
        # Kepler-only baseline
        print("\n📡 Training Kepler-Only Baseline...")
        kepler_baseline = self._train_single_telescope_model(
            self.kepler_df, 'Kepler Baseline'
        )
        baselines['kepler_only'] = kepler_baseline
        
        # TESS-only baseline
        print("\n📡 Training TESS-Only Baseline...")
        tess_baseline = self._train_single_telescope_model(
            self.tess_df, 'TESS Baseline'
        )
        baselines['tess_only'] = tess_baseline
        
        self.results['baselines'] = baselines
        return baselines
    
    def _train_single_telescope_model(self, data, model_name):
        """Train and evaluate a model on single telescope data"""
        # Prepare features and target
        X = data[self.feature_cols].copy()
        y = data[self.target_col].copy()
        
        # Fill missing values with median BEFORE splitting
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model (using HistGradientBoosting which handles NaN natively)
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=15,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"  ✅ {model_name}")
        print(f"     Training samples: {len(X_train)}")
        print(f"     Test samples: {len(X_test)}")
        print(f"     Accuracy: {accuracy:.4f}")
        print(f"     F1 Score: {f1:.4f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'accuracy': accuracy,
            'f1_score': f1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'predictions': y_pred,
            'true_labels': y_test,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def domain_adaptation_mixed_training(self):
        """Train on 80% Kepler + 20% TESS, test on both"""
        print("\n" + "=" * 70)
        print("DOMAIN ADAPTATION: MIXED TRAINING (80% Kepler + 20% TESS)")
        print("=" * 70)
        
        # Sample 80% Kepler and 20% TESS for training
        kepler_train_size = int(len(self.kepler_df) * 0.8)
        tess_train_size = int(kepler_train_size * 0.25)  # 20% of training set
        
        # Split Kepler data
        kepler_shuffled = self.kepler_df.sample(frac=1, random_state=42)
        kepler_train = kepler_shuffled.iloc[:kepler_train_size]
        kepler_test = kepler_shuffled.iloc[kepler_train_size:]
        
        # Split TESS data
        tess_shuffled = self.tess_df.sample(frac=1, random_state=42)
        tess_train = tess_shuffled.iloc[:tess_train_size]
        tess_test = tess_shuffled.iloc[tess_train_size:]
        
        # Combine training data
        train_data = pd.concat([kepler_train, tess_train], ignore_index=True)
        train_data = train_data.sample(frac=1, random_state=42)  # Shuffle
        
        print(f"\n📊 Training Set Composition:")
        print(f"  Kepler: {len(kepler_train)} ({len(kepler_train)/len(train_data)*100:.1f}%)")
        print(f"  TESS: {len(tess_train)} ({len(tess_train)/len(train_data)*100:.1f}%)")
        print(f"  Total: {len(train_data)}")
        
        # Prepare features
        X_train = train_data[self.feature_cols].copy()
        y_train = train_data[self.target_col]
        
        # Fill missing values with median
        for col in X_train.columns:
            if X_train[col].isna().any():
                X_train[col] = X_train[col].fillna(X_train[col].median())
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model (using HistGradientBoosting which handles NaN natively)
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=15,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Test on Kepler
        print("\n📡 Testing on Kepler Data:")
        kepler_results = self._evaluate_on_test_set(
            model, scaler, kepler_test, "Kepler Test Set"
        )
        
        # Test on TESS
        print("\n📡 Testing on TESS Data:")
        tess_results = self._evaluate_on_test_set(
            model, scaler, tess_test, "TESS Test Set"
        )
        
        self.results['mixed_training'] = {
            'model': model,
            'scaler': scaler,
            'kepler_results': kepler_results,
            'tess_results': tess_results,
            'train_composition': {
                'kepler': len(kepler_train),
                'tess': len(tess_train)
            }
        }
        
        return self.results['mixed_training']
    
    def reverse_transfer_experiments(self):
        """Test reverse transfer: Train on TESS, test on Kepler and vice versa"""
        print("\n" + "=" * 70)
        print("REVERSE TRANSFER EXPERIMENTS")
        print("=" * 70)
        
        reverse_results = {}
        
        # Experiment 1: Train on Kepler, test on TESS
        print("\n🔄 Experiment 1: Train on Kepler → Test on TESS")
        kepler_to_tess = self._cross_telescope_transfer(
            self.kepler_df, self.tess_df, "Kepler→TESS"
        )
        reverse_results['kepler_to_tess'] = kepler_to_tess
        
        # Experiment 2: Train on TESS, test on Kepler
        print("\n🔄 Experiment 2: Train on TESS → Test on Kepler")
        tess_to_kepler = self._cross_telescope_transfer(
            self.tess_df, self.kepler_df, "TESS→Kepler"
        )
        reverse_results['tess_to_kepler'] = tess_to_kepler
        
        self.results['reverse_transfer'] = reverse_results
        return reverse_results
    
    def _cross_telescope_transfer(self, source_data, target_data, experiment_name):
        """Train on source telescope, test on target telescope"""
        # Prepare source data (80/20 split for validation)
        X_source = source_data[self.feature_cols].copy()
        y_source = source_data[self.target_col]
        
        # Fill missing values with median
        for col in X_source.columns:
            if X_source[col].isna().any():
                X_source[col] = X_source[col].fillna(X_source[col].median())
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_source, y_source, test_size=0.2, random_state=42, stratify=y_source
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model (using HistGradientBoosting which handles NaN natively)
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=15,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Validate on source telescope
        y_val_pred = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f"\n  📊 Source Telescope Validation:")
        print(f"     Training samples: {len(X_train)}")
        print(f"     Validation samples: {len(X_val)}")
        print(f"     Accuracy: {val_accuracy:.4f}")
        print(f"     F1 Score: {val_f1:.4f}")
        
        # Test on target telescope
        print(f"\n  🎯 Target Telescope Test:")
        target_results = self._evaluate_on_test_set(
            model, scaler, target_data, experiment_name
        )
        
        return {
            'model': model,
            'scaler': scaler,
            'source_val_accuracy': val_accuracy,
            'source_val_f1': val_f1,
            'target_results': target_results,
            'train_size': len(X_train),
            'val_size': len(X_val)
        }
    
    def _evaluate_on_test_set(self, model, scaler, test_data, name):
        """Evaluate model on test set"""
        X_test = test_data[self.feature_cols].copy()
        y_test = test_data[self.target_col]
        
        # Fill missing values with median
        for col in X_test.columns:
            if X_test[col].isna().any():
                X_test[col] = X_test[col].fillna(X_test[col].median())
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"     Test samples: {len(X_test)}")
        print(f"     Accuracy: {accuracy:.4f}")
        print(f"     F1 Score: {f1:.4f}")
        
        # Class distribution
        print(f"\n     Class Distribution:")
        for cls in sorted(y_test.unique()):
            count = (y_test == cls).sum()
            pct = count / len(y_test) * 100
            print(f"       {cls}: {count} ({pct:.1f}%)")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'true_labels': y_test,
            'test_size': len(X_test),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def generate_summary_report(self):
        """Generate comprehensive summary of all experiments"""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE SUMMARY REPORT")
        print("=" * 70)
        
        # Baseline comparison
        print("\n📊 BASELINE MODELS (Telescope-Specific):")
        print("-" * 70)
        if 'baselines' in self.results:
            for name, result in self.results['baselines'].items():
                print(f"\n  {name.replace('_', ' ').title()}:")
                print(f"    Training Size: {result['train_size']}")
                print(f"    Test Size: {result['test_size']}")
                print(f"    Accuracy: {result['accuracy']:.4f}")
                print(f"    F1 Score: {result['f1_score']:.4f}")
        
        # Mixed training results
        print("\n\n📊 DOMAIN ADAPTATION (Mixed Training):")
        print("-" * 70)
        if 'mixed_training' in self.results:
            mt = self.results['mixed_training']
            print(f"\n  Training Composition:")
            print(f"    Kepler: {mt['train_composition']['kepler']}")
            print(f"    TESS: {mt['train_composition']['tess']}")
            
            print(f"\n  Performance on Kepler Test Set:")
            print(f"    Accuracy: {mt['kepler_results']['accuracy']:.4f}")
            print(f"    F1 Score: {mt['kepler_results']['f1_score']:.4f}")
            
            print(f"\n  Performance on TESS Test Set:")
            print(f"    Accuracy: {mt['tess_results']['accuracy']:.4f}")
            print(f"    F1 Score: {mt['tess_results']['f1_score']:.4f}")
        
        # Reverse transfer results
        print("\n\n📊 REVERSE TRANSFER LEARNING:")
        print("-" * 70)
        if 'reverse_transfer' in self.results:
            rt = self.results['reverse_transfer']
            
            print(f"\n  Kepler → TESS Transfer:")
            k2t = rt['kepler_to_tess']
            print(f"    Source (Kepler) Validation - Accuracy: {k2t['source_val_accuracy']:.4f}, F1: {k2t['source_val_f1']:.4f}")
            print(f"    Target (TESS) Test - Accuracy: {k2t['target_results']['accuracy']:.4f}, F1: {k2t['target_results']['f1_score']:.4f}")
            print(f"    Transfer Gap: {(k2t['source_val_accuracy'] - k2t['target_results']['accuracy'])*100:.2f}%")
            
            print(f"\n  TESS → Kepler Transfer:")
            t2k = rt['tess_to_kepler']
            print(f"    Source (TESS) Validation - Accuracy: {t2k['source_val_accuracy']:.4f}, F1: {t2k['source_val_f1']:.4f}")
            print(f"    Target (Kepler) Test - Accuracy: {t2k['target_results']['accuracy']:.4f}, F1: {t2k['target_results']['f1_score']:.4f}")
            print(f"    Transfer Gap: {(t2k['source_val_accuracy'] - t2k['target_results']['accuracy'])*100:.2f}%")
        
        # Key insights
        self._print_key_insights()
        
        return self.results
    
    def _print_key_insights(self):
        """Print key insights from all experiments"""
        print("\n\n🔑 KEY INSIGHTS:")
        print("=" * 70)
        
        if 'distribution_analysis' in self.results:
            n_sig = (self.results['distribution_analysis']['ks_pvalue'] < 0.05).sum()
            n_total = len(self.results['distribution_analysis'])
            print(f"\n1. Feature Distribution Shift:")
            print(f"   {n_sig}/{n_total} features show significant distribution differences")
            print(f"   This indicates substantial domain shift between telescopes")
        
        if 'reverse_transfer' in self.results:
            rt = self.results['reverse_transfer']
            k2t_gap = rt['kepler_to_tess']['source_val_accuracy'] - rt['kepler_to_tess']['target_results']['accuracy']
            t2k_gap = rt['tess_to_kepler']['source_val_accuracy'] - rt['tess_to_kepler']['target_results']['accuracy']
            
            print(f"\n2. Transfer Learning Performance:")
            print(f"   Kepler→TESS transfer gap: {k2t_gap*100:.2f}%")
            print(f"   TESS→Kepler transfer gap: {t2k_gap*100:.2f}%")
            
            if abs(k2t_gap) > abs(t2k_gap):
                print(f"   → TESS generalizes better to Kepler than vice versa")
            else:
                print(f"   → Kepler generalizes better to TESS than vice versa")
        
        if 'mixed_training' in self.results:
            mt = self.results['mixed_training']
            kepler_acc = mt['kepler_results']['accuracy']
            tess_acc = mt['tess_results']['accuracy']
            
            print(f"\n3. Mixed Training Benefits:")
            print(f"   Adding 20% TESS data helps balance telescope bias")
            print(f"   Kepler test accuracy: {kepler_acc:.4f}")
            print(f"   TESS test accuracy: {tess_acc:.4f}")
            print(f"   Difference: {abs(kepler_acc - tess_acc)*100:.2f}%")
    
    def save_results(self, output_file='telescope_domain_adaptation_results.csv'):
        """Save all results to CSV"""
        records = []
        
        # Baseline results
        if 'baselines' in self.results:
            for name, result in self.results['baselines'].items():
                records.append({
                    'experiment': name,
                    'train_telescope': name.split('_')[0],
                    'test_telescope': name.split('_')[0],
                    'train_size': result['train_size'],
                    'test_size': result['test_size'],
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score']
                })
        
        # Mixed training results
        if 'mixed_training' in self.results:
            mt = self.results['mixed_training']
            records.append({
                'experiment': 'mixed_training',
                'train_telescope': 'mixed',
                'test_telescope': 'kepler',
                'train_size': sum(mt['train_composition'].values()),
                'test_size': mt['kepler_results']['test_size'],
                'accuracy': mt['kepler_results']['accuracy'],
                'f1_score': mt['kepler_results']['f1_score']
            })
            records.append({
                'experiment': 'mixed_training',
                'train_telescope': 'mixed',
                'test_telescope': 'tess',
                'train_size': sum(mt['train_composition'].values()),
                'test_size': mt['tess_results']['test_size'],
                'accuracy': mt['tess_results']['accuracy'],
                'f1_score': mt['tess_results']['f1_score']
            })
        
        # Reverse transfer results
        if 'reverse_transfer' in self.results:
            rt = self.results['reverse_transfer']
            
            k2t = rt['kepler_to_tess']
            records.append({
                'experiment': 'kepler_to_tess',
                'train_telescope': 'kepler',
                'test_telescope': 'tess',
                'train_size': k2t['train_size'],
                'test_size': k2t['target_results']['test_size'],
                'accuracy': k2t['target_results']['accuracy'],
                'f1_score': k2t['target_results']['f1_score']
            })
            
            t2k = rt['tess_to_kepler']
            records.append({
                'experiment': 'tess_to_kepler',
                'train_telescope': 'tess',
                'test_telescope': 'kepler',
                'train_size': t2k['train_size'],
                'test_size': t2k['target_results']['test_size'],
                'accuracy': t2k['target_results']['accuracy'],
                'f1_score': t2k['target_results']['f1_score']
            })
        
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to {output_file}")
        
        return df


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("TELESCOPE DOMAIN ADAPTATION ANALYSIS")
    print("Analyzing transfer learning between Kepler and TESS missions")
    print("=" * 70)
    
    # Load data (assuming CSV files are available)
    try:
        print("\n📂 Loading datasets...")
        kepler_data = pd.read_csv('csvs/cumulative_2025.09.30_23.45.15.csv')
        tess_data = pd.read_csv('csvs/TOI_2025.09.30_23.45.34.csv')
        print(f"✅ Kepler data: {kepler_data.shape}")
        print(f"✅ TESS data: {tess_data.shape}")
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find data files")
        print("Please ensure data files are in the 'csvs' directory")
        return
    
    # Initialize adapter
    adapter = TelescopeDomainAdapter(kepler_data, tess_data)
    
    # Run analysis pipeline
    adapter.prepare_data()
    adapter.analyze_feature_distributions()
    adapter.train_baseline_models()
    adapter.domain_adaptation_mixed_training()
    adapter.reverse_transfer_experiments()
    adapter.generate_summary_report()
    
    # Save results
    results_df = adapter.save_results()
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nResults Summary:")
    print(results_df.to_string(index=False))
    

if __name__ == '__main__':
    main()
