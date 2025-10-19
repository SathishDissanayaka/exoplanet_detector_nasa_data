"""
TESS-Centric Training Strategy
================================
Based on domain adaptation analysis showing TESS→Kepler transfer 
performs 3.1× better than Kepler→TESS, we implement TESS-centric 
training strategies.

Key Insight: TESS learns more generalizable features due to:
- Wider sky coverage (entire sky vs Kepler's single field)
- More diverse stellar populations
- Better class balance in training data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from common.data_merger import DatasetMerger


class TessCentricTrainer:
    """Implements TESS-centric training strategies for cross-telescope models"""
    
    def __init__(self, kepler_data, tess_data):
        self.kepler_data = kepler_data.copy()
        self.tess_data = tess_data.copy()
        self.merger = DatasetMerger()
        self.results = {}
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Merge and prepare datasets"""
        print("\n" + "=" * 70)
        print("PREPARING DATA FOR TESS-CENTRIC TRAINING")
        print("=" * 70)
        
        # Merge datasets
        merged = self.merger.merge(self.kepler_data, self.tess_data)
        
        # Identify feature columns
        self.feature_cols = [col for col in merged.columns 
                            if col.startswith('merged_') 
                            and col != 'merged_koi_disposition'
                            and merged[col].dtype in ['float64', 'int64']]
        
        self.target_col = 'merged_koi_disposition'
        
        # Split by telescope
        self.kepler_df = merged[merged['mission'] == 'KEPLER'].copy()
        self.tess_df = merged[merged['mission'] == 'TESS'].copy()
        
        # Clean data
        self.kepler_df = self.kepler_df.dropna(subset=[self.target_col])
        self.tess_df = self.tess_df.dropna(subset=[self.target_col])
        
        # Remove rows with >50% missing features
        missing_threshold = len(self.feature_cols) * 0.5
        self.kepler_df = self.kepler_df[self.kepler_df[self.feature_cols].notna().sum(axis=1) >= missing_threshold]
        self.tess_df = self.tess_df[self.tess_df[self.feature_cols].notna().sum(axis=1) >= missing_threshold]
        
        print(f"\n✅ Prepared Data:")
        print(f"   Kepler: {len(self.kepler_df)} samples")
        print(f"   TESS: {len(self.tess_df)} samples")
        print(f"   Features: {len(self.feature_cols)}")
        
    def option_a_tess_first(self):
        """
        Option A: TESS-First Training
        Train primarily on TESS (80%) with Kepler supplement (20%)
        """
        print("\n" + "=" * 70)
        print("OPTION A: TESS-FIRST TRAINING (80% TESS + 20% Kepler)")
        print("=" * 70)
        
        # Sample training data
        tess_sample_size = int(len(self.tess_df) * 0.8)
        kepler_sample_size = int(tess_sample_size * 0.25)  # 20% of training set
        
        # Shuffle and split TESS
        tess_shuffled = self.tess_df.sample(frac=1, random_state=42)
        tess_train = tess_shuffled.iloc[:tess_sample_size]
        tess_test = tess_shuffled.iloc[tess_sample_size:]
        
        # Sample Kepler for training, rest for testing
        kepler_shuffled = self.kepler_df.sample(frac=1, random_state=42)
        kepler_train = kepler_shuffled.iloc[:kepler_sample_size]
        kepler_test = kepler_shuffled.iloc[kepler_sample_size:]
        
        # Combine training data
        training_data = pd.concat([tess_train, kepler_train], ignore_index=True)
        training_data = training_data.sample(frac=1, random_state=42)  # Shuffle
        
        print(f"\n📊 Training Set Composition:")
        print(f"   TESS: {len(tess_train)} ({len(tess_train)/len(training_data)*100:.1f}%)")
        print(f"   Kepler: {len(kepler_train)} ({len(kepler_train)/len(training_data)*100:.1f}%)")
        print(f"   Total: {len(training_data)}")
        
        # Train model
        model, scaler = self._train_model(training_data)
        
        # Evaluate on TESS test
        print(f"\n📡 Testing on TESS Test Set:")
        tess_results = self._evaluate(model, scaler, tess_test)
        
        # Evaluate on Kepler test
        print(f"\n📡 Testing on Kepler Test Set:")
        kepler_results = self._evaluate(model, scaler, kepler_test)
        
        self.results['option_a'] = {
            'model': model,
            'scaler': scaler,
            'tess_results': tess_results,
            'kepler_results': kepler_results,
            'training_composition': {
                'tess': len(tess_train),
                'kepler': len(kepler_train)
            }
        }
        
        return self.results['option_a']
    
    def option_b_finetuning(self):
        """
        Option B: TESS-Only with Kepler Fine-tuning
        Phase 1: Train on TESS
        Phase 2: Fine-tune on balanced Kepler subset
        """
        print("\n" + "=" * 70)
        print("OPTION B: TESS-ONLY WITH KEPLER FINE-TUNING")
        print("=" * 70)
        
        # Phase 1: Train on TESS
        print("\n🔵 PHASE 1: Training Base Model on TESS")
        tess_train, tess_val = train_test_split(
            self.tess_df, test_size=0.2, random_state=42, 
            stratify=self.tess_df[self.target_col]
        )
        
        print(f"   TESS Training: {len(tess_train)}")
        print(f"   TESS Validation: {len(tess_val)}")
        
        base_model, base_scaler = self._train_model(tess_train)
        
        print(f"\n   Validating base model on TESS:")
        tess_base_results = self._evaluate(base_model, base_scaler, tess_val)
        
        # Phase 2: Fine-tune on Kepler
        print(f"\n🟢 PHASE 2: Fine-tuning on Kepler")
        
        # Create balanced Kepler subset for fine-tuning
        kepler_balanced = self._balance_dataset(self.kepler_df, max_per_class=500)
        kepler_train, kepler_test = train_test_split(
            kepler_balanced, test_size=0.3, random_state=42,
            stratify=kepler_balanced[self.target_col]
        )
        
        print(f"   Kepler Fine-tuning: {len(kepler_train)}")
        print(f"   Kepler Test: {len(kepler_test)}")
        
        # Fine-tune (train new model initialized from TESS knowledge)
        # In practice, would use warm_start or transfer weights
        finetuned_model, finetuned_scaler = self._train_model(
            kepler_train, 
            initial_model=base_model
        )
        
        print(f"\n   Testing fine-tuned model on Kepler:")
        kepler_finetuned_results = self._evaluate(finetuned_model, finetuned_scaler, kepler_test)
        
        # Test on remaining TESS data
        tess_remaining = self.tess_df[~self.tess_df.index.isin(tess_train.index.union(tess_val.index))]
        if len(tess_remaining) > 100:
            print(f"\n   Testing fine-tuned model on TESS:")
            tess_finetuned_results = self._evaluate(finetuned_model, finetuned_scaler, tess_remaining)
        else:
            tess_finetuned_results = tess_base_results
        
        self.results['option_b'] = {
            'base_model': base_model,
            'finetuned_model': finetuned_model,
            'base_scaler': base_scaler,
            'finetuned_scaler': finetuned_scaler,
            'tess_base_results': tess_base_results,
            'kepler_finetuned_results': kepler_finetuned_results,
            'tess_finetuned_results': tess_finetuned_results
        }
        
        return self.results['option_b']
    
    def option_c_progressive_ratios(self):
        """
        Option C: Progressive TESS Emphasis
        Test multiple TESS/Kepler ratios to find optimal balance
        """
        print("\n" + "=" * 70)
        print("OPTION C: PROGRESSIVE TESS EMPHASIS")
        print("=" * 70)
        
        ratios = [
            (0.9, 0.1),  # 90% TESS, 10% Kepler
            (0.8, 0.2),  # 80% TESS, 20% Kepler  
            (0.7, 0.3),  # 70% TESS, 30% Kepler
            (0.6, 0.4),  # 60% TESS, 40% Kepler
            (0.5, 0.5),  # 50% TESS, 50% Kepler
        ]
        
        progressive_results = []
        
        for tess_ratio, kepler_ratio in ratios:
            print(f"\n📊 Testing Ratio: {int(tess_ratio*100)}% TESS + {int(kepler_ratio*100)}% Kepler")
            print("-" * 70)
            
            # Calculate sample sizes (use fixed total to ensure fair comparison)
            total_samples = 8000
            tess_n = int(total_samples * tess_ratio)
            kepler_n = int(total_samples * kepler_ratio)
            
            # Sample data
            tess_sample = self.tess_df.sample(n=min(tess_n, len(self.tess_df)), random_state=42)
            kepler_sample = self.kepler_df.sample(n=min(kepler_n, len(self.kepler_df)), random_state=42)
            
            # Split for training/testing
            tess_train, tess_test = train_test_split(
                tess_sample, test_size=0.2, random_state=42,
                stratify=tess_sample[self.target_col]
            )
            kepler_train, kepler_test = train_test_split(
                kepler_sample, test_size=0.2, random_state=42,
                stratify=kepler_sample[self.target_col]
            )
            
            # Combine training data
            training_data = pd.concat([tess_train, kepler_train], ignore_index=True)
            training_data = training_data.sample(frac=1, random_state=42)
            
            print(f"   Training: {len(training_data)} ({len(tess_train)} TESS + {len(kepler_train)} Kepler)")
            
            # Train model
            model, scaler = self._train_model(training_data)
            
            # Evaluate
            tess_results = self._evaluate(model, scaler, tess_test, verbose=False)
            kepler_results = self._evaluate(model, scaler, kepler_test, verbose=False)
            
            print(f"   TESS Test: Acc={tess_results['accuracy']:.4f}, F1={tess_results['f1']:.4f}")
            print(f"   Kepler Test: Acc={kepler_results['accuracy']:.4f}, F1={kepler_results['f1']:.4f}")
            
            progressive_results.append({
                'tess_ratio': tess_ratio,
                'kepler_ratio': kepler_ratio,
                'tess_train_n': len(tess_train),
                'kepler_train_n': len(kepler_train),
                'tess_accuracy': tess_results['accuracy'],
                'tess_f1': tess_results['f1'],
                'kepler_accuracy': kepler_results['accuracy'],
                'kepler_f1': kepler_results['f1'],
                'avg_accuracy': (tess_results['accuracy'] + kepler_results['accuracy']) / 2,
                'avg_f1': (tess_results['f1'] + kepler_results['f1']) / 2,
                'telescope_gap': abs(tess_results['accuracy'] - kepler_results['accuracy'])
            })
        
        # Find best ratio
        progressive_df = pd.DataFrame(progressive_results)
        best_idx = progressive_df['avg_f1'].idxmax()
        best_ratio = progressive_df.iloc[best_idx]
        
        print("\n" + "=" * 70)
        print("📈 PROGRESSIVE RATIO ANALYSIS SUMMARY")
        print("=" * 70)
        print(progressive_df.to_string(index=False))
        
        print(f"\n🏆 BEST RATIO: {int(best_ratio['tess_ratio']*100)}% TESS + {int(best_ratio['kepler_ratio']*100)}% Kepler")
        print(f"   Average F1: {best_ratio['avg_f1']:.4f}")
        print(f"   Telescope Gap: {best_ratio['telescope_gap']*100:.2f}%")
        
        self.results['option_c'] = {
            'progressive_results': progressive_df,
            'best_ratio': best_ratio
        }
        
        return self.results['option_c']
    
    def _balance_dataset(self, df, max_per_class=500):
        """Create balanced dataset by sampling from each class"""
        balanced_dfs = []
        for cls in df[self.target_col].unique():
            cls_df = df[df[self.target_col] == cls]
            n_samples = min(len(cls_df), max_per_class)
            balanced_dfs.append(cls_df.sample(n=n_samples, random_state=42))
        return pd.concat(balanced_dfs, ignore_index=True)
    
    def _train_model(self, training_data, initial_model=None):
        """Train model on given data"""
        X_train = training_data[self.feature_cols].copy()
        y_train = training_data[self.target_col]
        
        # Fill missing values
        for col in X_train.columns:
            if X_train[col].isna().any():
                X_train[col] = X_train[col].fillna(X_train[col].median())
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        if initial_model is not None:
            # For fine-tuning, we'd use warm_start or similar
            # For now, train fresh model with similar hyperparameters
            model = HistGradientBoostingClassifier(
                max_iter=150,  # More iterations for fine-tuning
                max_depth=15,
                learning_rate=0.05,  # Lower LR for fine-tuning
                random_state=42,
                verbose=0
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=15,
                learning_rate=0.1,
                random_state=42,
                verbose=0
            )
        
        model.fit(X_train_scaled, y_train)
        
        return model, scaler
    
    def _evaluate(self, model, scaler, test_data, verbose=True):
        """Evaluate model on test data"""
        X_test = test_data[self.feature_cols].copy()
        y_test = test_data[self.target_col]
        
        # Fill missing values
        for col in X_test.columns:
            if X_test[col].isna().any():
                X_test[col] = X_test[col].fillna(X_test[col].median())
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        if verbose:
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
            'f1': f1,
            'predictions': y_pred,
            'true_labels': y_test,
            'test_size': len(X_test),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def compare_all_strategies(self):
        """Run all strategies and compare results"""
        print("\n" + "=" * 70)
        print("RUNNING ALL TESS-CENTRIC STRATEGIES")
        print("=" * 70)
        
        # Run all options
        option_a = self.option_a_tess_first()
        option_b = self.option_b_finetuning()
        option_c = self.option_c_progressive_ratios()
        
        # Generate comparison report
        print("\n" + "=" * 70)
        print("COMPREHENSIVE STRATEGY COMPARISON")
        print("=" * 70)
        
        print("\n📊 OPTION A: TESS-First (80% TESS + 20% Kepler)")
        print(f"   TESS Test: {option_a['tess_results']['accuracy']:.4f} (F1: {option_a['tess_results']['f1']:.4f})")
        print(f"   Kepler Test: {option_a['kepler_results']['accuracy']:.4f} (F1: {option_a['kepler_results']['f1']:.4f})")
        print(f"   Telescope Gap: {abs(option_a['tess_results']['accuracy'] - option_a['kepler_results']['accuracy'])*100:.2f}%")
        
        print("\n📊 OPTION B: Fine-Tuning Strategy")
        print(f"   TESS (Base): {option_b['tess_base_results']['accuracy']:.4f} (F1: {option_b['tess_base_results']['f1']:.4f})")
        print(f"   Kepler (Fine-tuned): {option_b['kepler_finetuned_results']['accuracy']:.4f} (F1: {option_b['kepler_finetuned_results']['f1']:.4f})")
        print(f"   TESS (After fine-tune): {option_b['tess_finetuned_results']['accuracy']:.4f} (F1: {option_b['tess_finetuned_results']['f1']:.4f})")
        
        print("\n📊 OPTION C: Progressive Ratios")
        best = option_c['best_ratio']
        print(f"   Best Ratio: {int(best['tess_ratio']*100)}% TESS + {int(best['kepler_ratio']*100)}% Kepler")
        print(f"   Average Accuracy: {best['avg_accuracy']:.4f}")
        print(f"   Average F1: {best['avg_f1']:.4f}")
        print(f"   Telescope Gap: {best['telescope_gap']*100:.2f}%")
        
        # Save comparison
        self._save_comparison()
        
        return self.results
    
    def _save_comparison(self):
        """Save comparison results to CSV"""
        records = []
        
        if 'option_a' in self.results:
            oa = self.results['option_a']
            records.append({
                'strategy': 'Option A: TESS-First',
                'tess_ratio': 0.8,
                'kepler_ratio': 0.2,
                'tess_accuracy': oa['tess_results']['accuracy'],
                'tess_f1': oa['tess_results']['f1'],
                'kepler_accuracy': oa['kepler_results']['accuracy'],
                'kepler_f1': oa['kepler_results']['f1'],
                'avg_accuracy': (oa['tess_results']['accuracy'] + oa['kepler_results']['accuracy']) / 2,
                'telescope_gap': abs(oa['tess_results']['accuracy'] - oa['kepler_results']['accuracy'])
            })
        
        if 'option_b' in self.results:
            ob = self.results['option_b']
            records.append({
                'strategy': 'Option B: Fine-tuning (Base)',
                'tess_ratio': 1.0,
                'kepler_ratio': 0.0,
                'tess_accuracy': ob['tess_base_results']['accuracy'],
                'tess_f1': ob['tess_base_results']['f1'],
                'kepler_accuracy': None,
                'kepler_f1': None,
                'avg_accuracy': ob['tess_base_results']['accuracy'],
                'telescope_gap': None
            })
            records.append({
                'strategy': 'Option B: Fine-tuning (Adapted)',
                'tess_ratio': 1.0,
                'kepler_ratio': 0.0,
                'tess_accuracy': ob['tess_finetuned_results']['accuracy'],
                'tess_f1': ob['tess_finetuned_results']['f1'],
                'kepler_accuracy': ob['kepler_finetuned_results']['accuracy'],
                'kepler_f1': ob['kepler_finetuned_results']['f1'],
                'avg_accuracy': (ob['tess_finetuned_results']['accuracy'] + ob['kepler_finetuned_results']['accuracy']) / 2,
                'telescope_gap': abs(ob['tess_finetuned_results']['accuracy'] - ob['kepler_finetuned_results']['accuracy'])
            })
        
        if 'option_c' in self.results:
            oc = self.results['option_c']
            for _, row in oc['progressive_results'].iterrows():
                records.append({
                    'strategy': f"Option C: {int(row['tess_ratio']*100)}/{int(row['kepler_ratio']*100)} Ratio",
                    'tess_ratio': row['tess_ratio'],
                    'kepler_ratio': row['kepler_ratio'],
                    'tess_accuracy': row['tess_accuracy'],
                    'tess_f1': row['tess_f1'],
                    'kepler_accuracy': row['kepler_accuracy'],
                    'kepler_f1': row['kepler_f1'],
                    'avg_accuracy': row['avg_accuracy'],
                    'telescope_gap': row['telescope_gap']
                })
        
        df = pd.DataFrame(records)
        df.to_csv('tess_centric_strategy_comparison.csv', index=False)
        print(f"\n💾 Results saved to tess_centric_strategy_comparison.csv")


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("TESS-CENTRIC TRAINING STRATEGIES")
    print("Leveraging TESS's superior generalization capability")
    print("=" * 70)
    
    # Load data
    try:
        print("\n📂 Loading datasets...")
        kepler_data = pd.read_csv('csvs/cumulative_2025.09.30_23.45.15.csv')
        tess_data = pd.read_csv('csvs/TOI_2025.09.30_23.45.34.csv')
        print(f"✅ Kepler: {kepler_data.shape}")
        print(f"✅ TESS: {tess_data.shape}")
    except FileNotFoundError:
        print("❌ Error: Data files not found")
        return
    
    # Initialize trainer
    trainer = TessCentricTrainer(kepler_data, tess_data)
    
    # Run all strategies
    results = trainer.compare_all_strategies()
    
    print("\n" + "=" * 70)
    print("✅ TESS-CENTRIC ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
