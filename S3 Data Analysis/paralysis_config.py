# paralysis_config.py

import os
# from scipy.stats import uniform, randint # Not directly used here for Optuna defaults
import json  # For export/import
import config_paths

# Use config_paths for cross-platform compatibility
# Models are bundled resources, logs and analysis go to output directory
MODEL_DIR = str(config_paths.get_models_dir())
OUTPUT_BASE = str(config_paths.get_output_base_dir())
LOG_DIR = os.path.join(OUTPUT_BASE, 'logs')
ANALYSIS_DIR = os.path.join(OUTPUT_BASE, 'analysis_results')

# Create writable directories (not MODEL_DIR as it's bundled)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

CLASS_NAMES = {0: 'Normal', 1: 'Partial', 2: 'Complete'}  # Used by PARALYSIS_MAP in utils and pipeline
# Note: Using 'Normal' instead of 'None' because pandas interprets 'None' as NaN when reading CSV
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
}
INPUT_FILES = {
    # Use PyFaceAU-extracted data for training (matches inference pipeline)
    'results_csv': os.path.expanduser('~/Documents/SplitFace/S3O Results/combined_results.csv'),
    'expert_key_csv': os.path.join(os.path.dirname(__file__), 'FPRS FP Key.csv')
}

# These defaults will be applied to each zone and can be overridden.
ZONE_CONFIG_DEFAULTS = {
    'feature_extraction': {
        'use_normalized': True,
        'percent_diff_cap': 200.0,
        'min_value': 0.0001,
        'add_interaction_features': False,  # Disabled for simplicity and performance
        'add_statistical_features': False  # Disabled for simplicity and performance
    },
    'feature_selection': {  # This config is for the preliminary FS step *within* train_model_workflow
        'enabled': True,
        'top_n_features': 50,  # Default, can be overridden per zone
        # 'importance_file' is no longer used by train_model_workflow for selection.
        # The 'method' here is conceptual, actual method is RF-based in train_model_workflow.
        'method': 'rf_importance_in_workflow',
    },
    'training': {
        'test_size': 0.25,
        'random_state': 42,
        # 'early_stopping_rounds' for XGBoost is now handled inside Optuna (for its XGB eval)
        # or passed to the final XGBoost model if not using Optuna.
        'use_ensemble': True,  # This will translate to VotingClassifier in the trainer
        'model_params': {  # Default XGBoost parameters (base for Optuna search)
            # Objective, num_class, eval_metric will be automatically set based on num_classes.
            # 'objective': 'multi:softprob',
            # 'num_class': 3,
            'learning_rate': 0.05,  # A common starting point
            'max_depth': 5,
            'min_child_weight': 1,  # Often 1 is a good default
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'n_estimators': 300,  # Optuna will search over this
            # 'eval_metric': 'mlogloss',
            'tree_method': 'hist',  # Generally faster
            # 'scale_pos_weight' is not used here as sample_weight is passed to XGBoost.fit
        },
        'hyperparameter_tuning': {
            'enabled': True,
            'method': 'optuna',
            'optuna': {
                'n_trials': 100,
                'cv_folds': 5,
                'direction': 'maximize',
                'scoring': 'f1_macro',  # Single, robust objective
                'sampler': 'TPESampler',  # Default sampler
                'pruner': 'HyperbandPruner',  # Aggressive pruner
                'param_distributions': {  # For XGBoost base in VotingClassifier
                    'learning_rate': ['float', 0.005, 0.2, {'log': True}],
                    'max_depth': ['int', 3, 8],
                    'n_estimators': ['int', 100, 800],
                    'min_child_weight': ['int', 1, 7],
                    'gamma': ['float', 0.0, 0.4],
                    'subsample': ['float', 0.6, 1.0],
                    'colsample_bytree': ['float', 0.6, 1.0],
                    'reg_alpha': ['float', 1e-3, 1.0, {'log': True}],  # Added log and low for alpha
                    'reg_lambda': ['float', 1e-3, 2.0, {'log': True}],  # Added log and low for lambda
                    # 'scale_pos_weight' can be tuned if not using sample_weight directly
                },
                'optuna_early_stopping_rounds': 30,  # For XGBoost inside Optuna objective (fit method)
                'patience': 15  # For Optuna study early stopping (if enabled by Optuna version/callback)
            }
        },
        'smote': {
            'enabled': True,
            'variant': 'borderline',  # Good for minority classes near boundary
            'k_neighbors': 5,  # Standard default, adjust based on data size
            'sampling_strategy': 'adaptive',
            'adaptive_strategy_params': {
                'target_ratio_partial_to_majority': 0.8,  # Aim for 80% of majority for partial
                'target_ratio_complete_to_majority': 0.9,  # Aim for 90% of majority for complete
                'min_samples_after_smote': 75,  # Ensure a minimum number of samples
                'borderline_kind': 'borderline-1'  # Common BorderlineSMOTE variant
            },
            'apply_per_fold_in_tuning': True,  # Crucial for robust Optuna results
            'min_samples_per_class': 50,  # For SMOTE's internal k_neighbors check and adaptive strategy
            'use_smoteenn_after': True,  # SMOTE then ENN cleaning
            'use_tomek_after': False,  # SMOTEENN preferred over SMOTE + Tomek
            'enn_sampling_strategy': 'auto',  # ENN default
            'enn_kind_sel': 'mode'  # ENN default
        },
        'calibration': {
            'method': 'isotonic',  # Generally good for non-parametric calibration
            'cv': 5,  # Integer CV for CalibratedClassifierCV
            'ensemble': True,  # Conceptual, CalibratedClassifierCV wraps the (voting) ensemble
            'n_jobs': -1  # Use available cores for calibration CV
        },
        'class_weights': {  # Used for XGBoost sample_weight and RF/ET class_weight param
            # These are example weights, should be tuned based on class imbalance and importance
            0: 1.0,  # None
            1: 3.0,  # Partial (higher weight due to importance/imbalance)
            2: 2.0  # Complete
        },
        'threshold_optimization': {
            'enabled': True,
            'method': 'f1_maximize',  # Optimize for F1-score of each class (one-vs-rest style)
            'partial_class_range': [0.15, 0.6],  # Wider search range for 'Partial' class threshold
            'step_size': 0.02  # Finer step size for threshold search
        },
        'review_analysis': {  # Settings for generating review candidates
            'enabled': True,
            'top_k_influence': 50,  # Placeholder, influence analysis is complex
            'entropy_quantile': 0.9,  # Flag top 10% highest entropy samples
            'margin_quantile': 0.1,  # Flag bottom 10% lowest margin samples
            'true_label_prob_threshold': 0.4  # Flag if true label prob is below this
        },
        'ordinal_classification': {  # Ordinal approach for None < Partial < Complete
            'enabled': True,  # Enable ordinal classification by default
            'method': 'cumulative_binary',  # Binary decomposition approach
            'thresholds': {
                'threshold_1': 0.5,  # P(Y > 0) threshold: None vs (Partial+Complete)
                'threshold_2': 0.5   # P(Y > 1) threshold: (None+Partial) vs Complete
            },
            'optimize_thresholds': True,  # Search for optimal thresholds via validation
            'threshold_search_range': [0.2, 0.8],
            'threshold_step_size': 0.05,
            'use_ensemble': True,  # Use VotingClassifier for each binary model
            'class_weights_binary': {  # Weights for binary classifiers
                'model_1': {0: 1.0, 1: 2.0},  # None vs Affected
                'model_2': {0: 1.5, 1: 1.0}   # Non-Complete vs Complete
            }
        }
    }
}

ZONE_CONFIG = {
    'lower': {
        'name': 'Lower Face',
        'actions': ['BS', 'SS', 'SO', 'SE'],  # Example actions
        # AU16_r excluded - not available in OpenFace 2.2 / PyFaceAU
        'aus': ['AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r'],
        # Example AUs
        'expert_columns': {  # From original config
            'left': 'Paralysis - Left Lower Face',
            'right': 'Paralysis - Right Lower Face'
        },
        'target_columns': {  # Not directly used by trainer but good for context
            'left': 'Target_Left_Lower',
            'right': 'Target_Right_Lower'
        },
        'filenames': {
            'model': os.path.join(MODEL_DIR, 'lower_face_model.pkl'),
            'scaler': os.path.join(MODEL_DIR, 'lower_face_scaler.pkl'),
            'feature_list': os.path.join(MODEL_DIR, 'lower_face_features.list'),  # Will store selected features
            'importance': os.path.join(MODEL_DIR, 'lower_face_feature_importance.csv'),
            'training_log': os.path.join(LOG_DIR, 'lower_face_training.log'),
            'analysis_log': os.path.join(ANALYSIS_DIR, 'lowerface_analysis.log'),
            'critical_errors_report': os.path.join(ANALYSIS_DIR, 'lowerface_critical_errors_report.txt'),
            # Fixed extension
            'partial_errors_report': os.path.join(ANALYSIS_DIR, 'lowerface_partial_errors_report.txt'),
            # Fixed extension
            'review_candidates_csv': os.path.join(ANALYSIS_DIR, 'lowerface_review_candidates.csv')
        },
        # These will be filled by the default merging logic below
    },
    'mid': {
        'name': 'Mid Face',
        'actions': ['ES', 'ET', 'BK'],
        'aus': ['AU45_r', 'AU07_r', 'AU06_r'],  # AU06 needed for best performance
        'expert_columns': {
            'left': 'Paralysis - Left Mid Face',
            'right': 'Paralysis - Right Mid Face'
        },
        'target_columns': {
            'left': 'Target_Left_Mid',
            'right': 'Target_Right_Mid'
        },
        'filenames': {
            'model': os.path.join(MODEL_DIR, 'mid_face_model.pkl'),
            'scaler': os.path.join(MODEL_DIR, 'mid_face_scaler.pkl'),
            'feature_list': os.path.join(MODEL_DIR, 'mid_face_features.list'),
            'importance': os.path.join(MODEL_DIR, 'mid_face_feature_importance.csv'),
            'training_log': os.path.join(LOG_DIR, 'mid_face_training.log'),
            'analysis_log': os.path.join(ANALYSIS_DIR, 'midface_analysis.log'),
            'critical_errors_report': os.path.join(ANALYSIS_DIR, 'midface_critical_errors_report.txt'),
            'partial_errors_report': os.path.join(ANALYSIS_DIR, 'midface_partial_errors_report.txt'),
            'review_candidates_csv': os.path.join(ANALYSIS_DIR, 'midface_review_candidates.csv')
        },
        # OPTIMIZED V2: Mid face - more aggressive settings for extreme imbalance
        # 75% None, 15% Partial, 10% Complete
        'training': {
            'class_weights': {
                0: 1.0,   # None (majority)
                1: 10.0,  # Partial (very heavily weighted for better recall)
                2: 7.0    # Complete (heavily weighted)
            },
            'smote': {
                'enabled': True,
                'variant': 'regular',  # Use regular SMOTE (not borderline) for small minority
                'k_neighbors': 3,  # Fewer neighbors for very small classes
                'sampling_strategy': 'adaptive',
                'adaptive_strategy_params': {
                    'target_ratio_partial_to_majority': 0.7,  # More aggressive oversampling
                    'target_ratio_complete_to_majority': 0.6,  # More aggressive oversampling
                    'min_samples_after_smote': 50,
                    'borderline_kind': 'borderline-1'
                },
                'apply_per_fold_in_tuning': True,
                'min_samples_per_class': 30,  # Lower threshold for small classes
                'use_smoteenn_after': False,  # DISABLE SMOTEENN - preserve minority samples
                'use_tomek_after': False
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'method': 'optuna',
                'optuna': {
                    'n_trials': 200,
                    'cv_folds': 5,
                    'direction': 'maximize',
                    'scoring': 'f1_macro',
                    'sampler': 'TPESampler',
                    'pruner': 'HyperbandPruner',
                    'param_distributions': {
                        'learning_rate': ['float', 0.01, 0.15, {'log': True}],  # Narrower range
                        'max_depth': ['int', 3, 6],  # Shallower trees to prevent overfitting
                        'n_estimators': ['int', 100, 500],
                        'min_child_weight': ['int', 1, 5],
                        'gamma': ['float', 0.0, 0.3],
                        'subsample': ['float', 0.7, 1.0],
                        'colsample_bytree': ['float', 0.7, 1.0],
                        'reg_alpha': ['float', 0.01, 1.0, {'log': True}],
                        'reg_lambda': ['float', 0.01, 2.0, {'log': True}],
                    },
                    'optuna_early_stopping_rounds': 25,
                    'patience': 15
                },
                # OPTIMAL PARAMS from 92.59% accuracy run (2025-12-30 21:37:13)
                # Use these with 'use_known_optimal': True to skip Optuna
                'known_optimal_params': {
                    'learning_rate': 0.04926,
                    'max_depth': 3,
                    'n_estimators': 372,
                    'min_child_weight': 2,
                    'gamma': 0.1324,
                    'subsample': 0.742,
                    'colsample_bytree': 0.705,
                    'reg_alpha': 0.1589,
                    'reg_lambda': 0.2426
                },
                'use_known_optimal': False  # Set True to skip Optuna and use known params
            }
        },
        'feature_selection': {
            'enabled': True,
            'top_n_features': 40,  # Fewer features to avoid overfitting
            'method': 'rf_importance_in_workflow'
        }
    },
    'upper': {
        'name': 'Upper Face',
        'actions': ['RE'],
        'aus': ['AU01_r', 'AU02_r'],
        'expert_columns': {
            'left': 'Paralysis - Left Upper Face',
            'right': 'Paralysis - Right Upper Face'
        },
        'target_columns': {
            'left': 'Target_Left_Upper',
            'right': 'Target_Right_Upper'
        },
        'filenames': {
            'model': os.path.join(MODEL_DIR, 'upper_face_model.pkl'),
            'scaler': os.path.join(MODEL_DIR, 'upper_face_scaler.pkl'),
            'feature_list': os.path.join(MODEL_DIR, 'upper_face_features.list'),
            'importance': os.path.join(MODEL_DIR, 'upper_face_feature_importance.csv'),
            'training_log': os.path.join(LOG_DIR, 'upper_face_training.log'),
            'analysis_log': os.path.join(ANALYSIS_DIR, 'upperface_analysis.log'),
            'critical_errors_report': os.path.join(ANALYSIS_DIR, 'upperface_critical_errors_report.txt'),
            'partial_errors_report': os.path.join(ANALYSIS_DIR, 'upperface_partial_errors_report.txt'),
            'review_candidates_csv': os.path.join(ANALYSIS_DIR, 'upperface_review_candidates.csv')
        },
    }
}

for zone_key_iter, zone_data_iter in ZONE_CONFIG.items():
    # Start with a deep copy of defaults for this zone
    current_zone_config = {
        'feature_extraction': ZONE_CONFIG_DEFAULTS['feature_extraction'].copy(),
        'feature_selection': ZONE_CONFIG_DEFAULTS['feature_selection'].copy(),
        'training': ZONE_CONFIG_DEFAULTS['training'].copy()  # Shallow copy for training initially
    }
    # Deep copy nested dicts within training (like optuna, smote, etc.)
    for training_key, training_val in ZONE_CONFIG_DEFAULTS['training'].items():
        if isinstance(training_val, dict):
            current_zone_config['training'][training_key] = training_val.copy()
            if training_key == 'hyperparameter_tuning' and 'optuna' in training_val:  # Special case for optuna
                current_zone_config['training']['hyperparameter_tuning']['optuna'] = training_val['optuna'].copy()
            if training_key == 'smote' and 'adaptive_strategy_params' in training_val:
                current_zone_config['training']['smote']['adaptive_strategy_params'] = training_val[
                    'adaptive_strategy_params'].copy()

    # Update with specific settings from ZONE_CONFIG definition
    for main_key in ['feature_extraction', 'feature_selection', 'training']:
        if main_key in zone_data_iter:  # If zone has this main key
            for sub_key, sub_val in zone_data_iter[main_key].items():
                if isinstance(sub_val, dict) and sub_key in current_zone_config[main_key] and isinstance(
                        current_zone_config[main_key][sub_key], dict):
                    current_zone_config[main_key][sub_key].update(sub_val)
                else:
                    current_zone_config[main_key][sub_key] = sub_val

    # Ensure all other top-level keys from ZONE_CONFIG (name, actions, aus, etc.) are preserved
    for key, val in zone_data_iter.items():
        if key not in ['feature_extraction', 'feature_selection', 'training']:
            current_zone_config[key] = val

    ZONE_CONFIG[zone_key_iter] = current_zone_config

# Specific overrides after defaults are applied (example: if 'lower' needs more FS features)
if 'lower' in ZONE_CONFIG:
    ZONE_CONFIG['lower']['feature_selection']['top_n_features'] = 60
    ZONE_CONFIG['lower']['training']['class_weights'] = {0: 1.0, 1: 3.5, 2: 2.5}
    ZONE_CONFIG['lower']['training']['hyperparameter_tuning']['optuna']['n_trials'] = 200

if 'mid' in ZONE_CONFIG:
    # OPTIMIZED: Use aggressive weights for extreme imbalance (75%/15%/10%)
    ZONE_CONFIG['mid']['feature_selection']['top_n_features'] = 40
    ZONE_CONFIG['mid']['training']['class_weights'] = {0: 1.0, 1: 10.0, 2: 7.0}
    ZONE_CONFIG['mid']['training']['hyperparameter_tuning']['optuna']['n_trials'] = 200

if 'upper' in ZONE_CONFIG:
    ZONE_CONFIG['upper']['feature_selection']['top_n_features'] = 25
    ZONE_CONFIG['upper']['training']['class_weights'] = {0: 1.0, 1: 3.0, 2: 2.0}
    ZONE_CONFIG['upper']['training']['hyperparameter_tuning']['optuna']['n_trials'] = 200
    ZONE_CONFIG['upper']['training']['calibration']['method'] = 'sigmoid'

REVIEW_CONFIG = {
    'similarity_threshold': 0.95,
    'consistency_checks': {
        'cross_zone': True,
        'temporal': True,
        'feature_based': True
    },
    'priority_weights': {  # Used by review candidate generation logic
        'confidence': 0.3,  # Lower confidence = higher priority
        'error_severity': 0.3,  # Critical > Partial > Standard
        'inconsistency': 0.2,  # If flagged by consistency checks
        'influence': 0.2  # If flagged as highly influential (negative influence)
    },
    'export_format': 'xlsx',
    'include_features': True,
    'max_similar_patients': 5,
    'validation': {  # For potential future automated review validation
        'quick_validation_folds': 3,
        'min_improvement_threshold': 0.01,
        'significance_level': 0.05
    },
    'change_limits': {  # For potential future automated correction suggestions
        'max_changes_per_tier': {1: 30, 2: 50, 3: 100, 4: 200},
        'max_distribution_shift': 0.05
    },
    'review_tiers': {  # Conceptual tiers for organizing review candidates
        1: {'name': 'Critical/High Confidence Errors', 'priority': 'highest'},
        2: {'name': 'Consistency Issues', 'priority': 'high'},
        3: {'name': 'High Uncertainty', 'priority': 'medium'},
        4: {'name': 'General Review', 'priority': 'low'}
    }
}

ADVANCED_TRAINING_CONFIG = {
    'cross_validation': {  # Not directly used by workflow if Optuna handles CV for HPT
        'enabled': False,
        'folds': 5,
        'shuffle': True,
        'stratified': True
    },
    'ensemble_options': {  # Now primarily for VotingClassifier
        'voting_type': 'soft',  # For VotingClassifier
        'weights': {  # For VotingClassifier base model weights - these are defaults
            'xgb': 0.6,  # XGBoost
            'rf': 0.2,  # RandomForest
            'et': 0.2  # ExtraTrees
        },
        'random_forest_params': {  # For RF base model in VotingClassifier
            'n_estimators': 100,
            'max_depth': None,  # Let it grow
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            # 'class_weight': 'balanced_subsample', # Now handled by zone_config.training.class_weights passed to constructor
            'n_jobs': -1,  # Use all cores for RF
            'oob_score': False,  # Can enable for debugging RF
            'bootstrap': True,
        },
        'extra_trees_params': {  # For ET base model in VotingClassifier
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            # 'class_weight': 'balanced_subsample', # Handled by zone_config.training.class_weights
            'n_jobs': -1,  # Use all cores for ET
            'bootstrap': False,  # ET default
        },
    },
    'calibration': {  # Default calibration settings if not in zone_config
        'method': 'isotonic',
        'cv': 5,  # Integer for CV within CalibratedClassifierCV
        'ensemble': True,  # Conceptual: indicates the ensemble model is being calibrated
    },
    'evaluation_metrics': [  # Comprehensive list of metrics to calculate
        'accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro', 'f1_per_class',
        'cohen_kappa', 'matthews_corrcoef', 'auc_per_class', 'confusion_matrix',
        'classification_report', 'precision_recall_curve'
    ],
    'monitoring': {
        'save_intermediate_results': True,  # Conceptual, specific saving handled by pipeline
        'plot_optimization_history': True,  # For Optuna plots
        'calculate_feature_importance': True,  # Enabled by default
        'generate_learning_curves': False,  # Can be enabled for detailed analysis, time-consuming
        'save_model_checkpoints': True,  # For saving Optuna study & final model
        'log_predictions': True  # For error analysis files
    },
    'post_processing': {
        'confidence_recalibration': True,  # Done by CalibratedClassifierCV
        'threshold_optimization': True,  # Enabled via zone_config
        'min_confidence_threshold': 0.3  # Not directly used by current model trainer logic
    }
}

DATA_AUGMENTATION_CONFIG = {
    'enabled': False,  # Complex augmentations (beyond SMOTE) disabled by default for simplicity
    'methods': {  # Kept for potential re-enablement, but not used if 'enabled' is False
        'noise_injection': {'enabled': False, 'noise_level': 0.02, 'noise_type': 'gaussian'},
        'feature_perturbation': {'enabled': False, 'perturbation_factor': 0.05, 'perturbation_probability': 0.3},
        'mixup': {'enabled': False, 'alpha': 0.2, 'probability': 0.5},
        'feature_dropout': {'enabled': False, 'dropout_rate': 0.1, 'probability': 0.3}
    },
    'augmentation_factor': 0.3,  # How much augmented data to add relative to applicable set
    'apply_to_minority_only': True  # Only augment minority classes if enabled
}

PERFORMANCE_CONFIG = {
    'xgboost_optimizations': {  # Passed to XGBoost constructor where applicable
        'use_gpu': False,  # Set True if GPU available and XGBoost built with GPU support
        'tree_method': 'hist',
        'predictor': 'auto',  # 'cpu_predictor' or 'gpu_predictor' if use_gpu is True
        'grow_policy': 'depthwise',  # Default
        'max_bins': 256,
        'single_precision_histogram': True  # Can speed up hist tree_method
    },
    'memory_optimization': {
        'enable_gc': True,  # Explicit garbage collection calls
        'cache_feature_importance': True,  # Conceptual, not a direct flag for a function
        'reduce_memory_usage': True  # Conceptual
    },
    'parallel_processing': {
        'n_jobs': -1,  # Default for sklearn components that support it (e.g. RF, ET, CalibCV)
        # Optuna study.optimize runs with n_jobs=4 parallel trials (implemented in model_trainer)
        'backend': 'threading',  # Default for many sklearn components
        'batch_size': 'auto'  # For operations that support batching (not common in this pipeline)
    }
}

def get_zone_config(zone):
    return ZONE_CONFIG.get(zone, {})


def get_all_zones():
    return list(ZONE_CONFIG.keys())


def get_model_path(zone):
    zone_config = get_zone_config(zone)
    return zone_config.get('filenames', {}).get('model', None)


def get_training_params(zone):
    zone_config = get_zone_config(zone)
    return zone_config.get('training', {})


def update_zone_config(zone, updates):  # Deep update logic
    if zone in ZONE_CONFIG:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(ZONE_CONFIG[zone].get(key), dict):
                current_level = ZONE_CONFIG[zone][key]
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict) and isinstance(current_level.get(sub_key), dict):
                        # Further deep update for nested dicts like 'optuna'
                        if sub_key in current_level and isinstance(current_level[sub_key], dict):
                            current_level[sub_key].update(sub_value)
                        else:
                            current_level[sub_key] = sub_value  # or deepcopy(sub_value)
                    else:
                        current_level[sub_key] = sub_value
            else:
                ZONE_CONFIG[zone][key] = value

def validate_config():
    errors = []
    for zone, config in ZONE_CONFIG.items():
        if 'name' not in config: errors.append(f"Zone {zone}: missing 'name' field")
        if 'training' not in config:
            errors.append(f"Zone {zone}: missing 'training' field")
        else:
            training = config['training']
            if training.get('hyperparameter_tuning', {}).get('enabled') and \
                    training.get('hyperparameter_tuning', {}).get('method') == 'optuna':
                optuna_cfg = training['hyperparameter_tuning']['optuna']
                if optuna_cfg.get('scoring') == 'multi_objective' and \
                        'multi_objective_weights' not in optuna_cfg:
                    # This warning is less relevant now we default to single objective
                    # errors.append(f"Zone {zone}: 'multi_objective_weights' missing for multi_objective scoring.")
                    pass
            if training.get('smote', {}).get('sampling_strategy') == 'adaptive' and \
                    'adaptive_strategy_params' not in training.get('smote', {}):
                errors.append(f"Zone {zone}: 'adaptive_strategy_params' missing for adaptive SMOTE.")
            if training.get('threshold_optimization', {}).get('enabled') and \
                    'partial_class_range' not in training.get('threshold_optimization', {}):
                errors.append(f"Zone {zone}: 'partial_class_range' missing for threshold optimization.")
            if training.get('smote', {}).get('enabled') and \
                    training.get('smote', {}).get('use_smoteenn_after') and \
                    training.get('smote', {}).get('variant', 'regular') not in ['regular', '', None]:
                zone_variant = training['smote']['variant']
                # errors.append(f"INFO Zone {zone}: SMOTE variant '{zone_variant}' will be overridden to 'regular' when 'use_smoteenn_after' is True.") # This is an FYI, not error
        filenames = config.get('filenames', {})
        for key, path in filenames.items():
            if not path: errors.append(f"Zone {zone}: empty path for {key}")
    return errors


validation_errors_on_load = validate_config()

def get_class_weights_for_zone(zone):
    zone_config = get_zone_config(zone)
    return zone_config.get('training', {}).get('class_weights', {0: 1.0, 1: 1.0, 2: 1.0})


def get_smote_config_for_zone(zone):
    zone_config = get_zone_config(zone)
    return zone_config.get('training', {}).get('smote', {})


def get_threshold_optimization_config(zone):
    zone_config = get_zone_config(zone)
    return zone_config.get('training', {}).get('threshold_optimization', {})


def get_feature_extraction_config(zone):
    zone_config = get_zone_config(zone)
    return zone_config.get('feature_extraction', {})


def get_feature_selection_config(zone):
    zone_config = get_zone_config(zone)
    return zone_config.get('feature_selection', {})


def get_performance_targets(zone):
    targets = {
        'lower': {'balanced_accuracy': 0.75, 'f1_partial': 0.55, 'overall_f1': 0.80},
        'mid': {'balanced_accuracy': 0.82, 'f1_partial': 0.55, 'overall_f1': 0.85},
        'upper': {'balanced_accuracy': 0.78, 'f1_partial': 0.55, 'overall_f1': 0.82}
    }
    return targets.get(zone, {'balanced_accuracy': 0.75, 'f1_partial': 0.50, 'overall_f1': 0.80})

def apply_preset(preset_name):
    presets = {
        'aggressive_partial': {
            'updates': {
                'training': {
                    'class_weights': {0: 1.0, 1: 5.0, 2: 2.0},  # Stronger weight for partial
                    'smote': {'adaptive_strategy_params': {'target_ratio_partial_to_majority': 1.2}},
                    'hyperparameter_tuning': {'optuna': {'scoring': 'f1_partial'}}  # Optimize directly for f1_partial
                }
            }
        },
        'balanced_f1': {
            'updates': {
                'training': {
                    'class_weights': {0: 1.0, 1: 2.5, 2: 2.0},
                    'hyperparameter_tuning': {'optuna': {'scoring': 'f1_macro'}}
                }
            }
        },
        'fast_dev': {  # For quick iterations during development
            'updates': {
                'feature_selection': {'top_n_features': 20},
                'training': {
                    'hyperparameter_tuning': {'optuna': {'n_trials': 10, 'cv_folds': 2}},
                    'model_params': {'n_estimators': 50}  # For XGBoost base
                }
            }
        },
        'high_quality': {  # For best possible results, longer training
            'updates': {
                'feature_selection': {'top_n_features': 75},  # More features if they prove useful
                'training': {
                    'hyperparameter_tuning': {'optuna': {'n_trials': 200, 'cv_folds': 5}},
                    'model_params': {'n_estimators': 500}  # Higher n_estimators for XGB
                }
            }
        }
    }
    if preset_name in presets:
        preset = presets[preset_name]
        for zone_key in ZONE_CONFIG:
            update_zone_config(zone_key, preset['updates'])
        print(f"Applied preset: {preset_name}")
        # Re-validate after applying preset
        current_validation_errors = validate_config()
        if current_validation_errors:
            print("Configuration validation errors/warnings after applying preset:")
            for error in current_validation_errors: print(f"  - {error}")
        return True
    else:
        print(f"Unknown preset: {preset_name}. Available presets: {list(presets.keys())}")
        return False

def export_config_to_file(filename='paralysis_config_backup.json'):
    config_export = {
        'ZONE_CONFIG': ZONE_CONFIG,
        'REVIEW_CONFIG': REVIEW_CONFIG,
        'ADVANCED_TRAINING_CONFIG': ADVANCED_TRAINING_CONFIG,
        'DATA_AUGMENTATION_CONFIG': DATA_AUGMENTATION_CONFIG,
        'PERFORMANCE_CONFIG': PERFORMANCE_CONFIG
    }
    try:
        with open(filename, 'w') as f:
            json.dump(config_export, f, indent=2)
        print(f"Configuration exported to {filename}")
    except Exception as e:
        print(f"Error exporting configuration: {e}")


def import_config_from_file(filename='paralysis_config_backup.json'):
    global ZONE_CONFIG, REVIEW_CONFIG, ADVANCED_TRAINING_CONFIG, DATA_AUGMENTATION_CONFIG, PERFORMANCE_CONFIG
    try:
        with open(filename, 'r') as f:
            config_import = json.load(f)
        ZONE_CONFIG = config_import.get('ZONE_CONFIG', ZONE_CONFIG)
        REVIEW_CONFIG = config_import.get('REVIEW_CONFIG', REVIEW_CONFIG)
        ADVANCED_TRAINING_CONFIG = config_import.get('ADVANCED_TRAINING_CONFIG', ADVANCED_TRAINING_CONFIG)
        DATA_AUGMENTATION_CONFIG = config_import.get('DATA_AUGMENTATION_CONFIG', DATA_AUGMENTATION_CONFIG)
        PERFORMANCE_CONFIG = config_import.get('PERFORMANCE_CONFIG', PERFORMANCE_CONFIG)
        print(f"Configuration imported from {filename}")
        # Re-validate after import
        new_validation_errors = validate_config()
        if new_validation_errors:
            print("Configuration validation errors/warnings after import:")
            for error in new_validation_errors: print(f"  - {error}")
        else:
            print("Configuration validation passed after import.")
        return True
    except Exception as e:
        print(f"Failed to import configuration from {filename}: {e}")
        return False

def get_config_summary():
    summary = {}
    for zone in ZONE_CONFIG:
        zone_config = ZONE_CONFIG[zone]
        training = zone_config.get('training', {})
        fs_conf = zone_config.get('feature_selection', {})
        optuna_conf = training.get('hyperparameter_tuning', {}).get('optuna', {})
        smote_conf = training.get('smote', {})
        summary[zone] = {
            'Name': zone_config.get('name'),
            'FS Top N': fs_conf.get('top_n_features', 'N/A'),
            'Ensemble': training.get('use_ensemble', False),  # True implies VotingClassifier
            'SMOTE Variant': smote_conf.get('variant', 'N/A'),
            'SMOTEENN': smote_conf.get('use_smoteenn_after', False),
            'Optuna Trials': optuna_conf.get('n_trials', 'N/A'),
            'Optuna Scoring': optuna_conf.get('scoring', 'N/A'),
            'Class Weights': training.get('class_weights', {}),
            'Threshold Opt': training.get('threshold_optimization', {}).get('enabled', False)
        }
    return summary


def print_config_summary():
    summary = get_config_summary()
    print("\n" + "=" * 80 + "\nCONFIGURATION SUMMARY\n" + "=" * 80)
    for zone, config_summary_data in summary.items():  # renamed config to config_summary_data
        print(f"\n{zone.upper()} FACE ({config_summary_data['Name']}):")  # Use config_summary_data
        print(f"  Feature Selection Top N: {config_summary_data['FS Top N']}")
        print(f"  Ensemble (Voting): {config_summary_data['Ensemble']}")
        smote_variant_sum = config_summary_data['SMOTE Variant']
        if config_summary_data['SMOTEENN'] and smote_variant_sum not in ['regular', '', None]:
            smote_variant_sum += " (will be 'regular' for SMOTEENN base)"
        print(f"  SMOTE: {smote_variant_sum} (SMOTEENN: {config_summary_data['SMOTEENN']})")
        print(
            f"  Optuna Trials: {config_summary_data['Optuna Trials']}, Scoring: {config_summary_data['Optuna Scoring']}")
        print(f"  Class Weights (for models): {config_summary_data['Class Weights']}")
        print(f"  Threshold Optimization: {config_summary_data['Threshold Opt']}")
    print("=" * 80 + "\n")

def enable_all_optimizations():
    for zone_key in ZONE_CONFIG:
        updates = {
            'training': {
                'smote': {'use_smoteenn_after': True, 'enabled': True},
                'threshold_optimization': {'enabled': True},
                'hyperparameter_tuning': {'enabled': True},
                'use_ensemble': True,  # VotingClassifier
            },
            'feature_selection': {'enabled': True}  # Preliminary FS in workflow
        }
        update_zone_config(zone_key, updates)
    global DATA_AUGMENTATION_CONFIG
    DATA_AUGMENTATION_CONFIG['enabled'] = False  # Keep complex augmentation off
    print("Selected optimizations enabled (SMOTE, ThreshOpt, HPT, Ensemble, FS). Complex Augmentation remains OFF.")


def disable_all_optimizations():  # For baseline comparison (e.g. single XGB, no SMOTE, no HPT)
    for zone_key in ZONE_CONFIG:
        updates = {
            'training': {
                'smote': {'enabled': False, 'use_smoteenn_after': False},
                'threshold_optimization': {'enabled': False},
                'hyperparameter_tuning': {'enabled': False},
                'use_ensemble': False,  # Train single XGBoost (or whatever is default without ensemble)
            },
            'feature_selection': {'enabled': False},  # Use all features from prepare_data
            'feature_extraction': {
                'add_interaction_features': False,
                'add_statistical_features': False
            }
        }
        update_zone_config(zone_key, updates)
    global DATA_AUGMENTATION_CONFIG
    DATA_AUGMENTATION_CONFIG['enabled'] = False
    print("All additional optimizations disabled. Basic model setup (likely single XGB without HPT).")

if __name__ == "__main__":
    if validation_errors_on_load:
        print("Initial Configuration validation errors/warnings:")
        for error in validation_errors_on_load: print(f"  - {error}")
    else:
        print("Initial Configuration validation passed!")
    print_config_summary()
    # Example of using a preset:
    # apply_preset('fast_dev')
    # print_config_summary()
    # export_config_to_file('config_fast_dev.json')