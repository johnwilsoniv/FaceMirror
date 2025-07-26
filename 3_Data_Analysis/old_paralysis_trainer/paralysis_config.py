# paralysis_config.py (v4 - Enhanced Configuration)

import os
from scipy.stats import uniform, randint

# --- Base Paths ---
MODEL_DIR = 'models'
LOG_DIR = 'logs'
ANALYSIS_DIR = 'analysis_results'

# --- Ensure Directories Exist ---
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# --- Global Settings ---
CLASS_NAMES = {0: 'None', 1: 'Partial', 2: 'Complete'}
LOGGING_CONFIG = {
    'level': 'INFO',  # Change to DEBUG for more detailed logs
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
}
INPUT_FILES = {
    'results_csv': 'combined_results.csv',
    'expert_key_csv': 'FPRS FP Key.csv'
}

# --- Zone-Specific Configurations ---
ZONE_CONFIG = {
    'lower': {
        'name': 'Lower Face',
        'actions': ['BS', 'SS', 'SO', 'SE'],
        'aus': ['AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r'],
        'expert_columns': {
            'left': 'Paralysis - Left Lower Face',
            'right': 'Paralysis - Right Lower Face'
        },
        'target_columns': {
            'left': 'Target_Left_Lower',
            'right': 'Target_Right_Lower'
        },
        'filenames': {
            'model': os.path.join(MODEL_DIR, 'lower_face_model.pkl'),
            'scaler': os.path.join(MODEL_DIR, 'lower_face_scaler.pkl'),
            'feature_list': os.path.join(MODEL_DIR, 'lower_face_features.list'),
            'importance': os.path.join(MODEL_DIR, 'lower_face_feature_importance.csv'),
            'training_log': os.path.join(LOG_DIR, 'lower_face_training.log'),
            'analysis_log': os.path.join(ANALYSIS_DIR, 'lowerface_analysis.log'),
            'critical_errors_report': os.path.join(ANALYSIS_DIR, 'lowerface_critical_errors.txt'),
            'partial_errors_report': os.path.join(ANALYSIS_DIR, 'lowerface_partial_errors.txt'),
            'review_candidates_csv': os.path.join(ANALYSIS_DIR, 'lowerface_review_candidates.csv')
        },
        'feature_extraction': {
            'use_normalized': True,
            'percent_diff_cap': 200.0,
            'min_value': 0.0001
        },
        'feature_selection': {
            'enabled': True,  # Lower face has many features, keep them all
            'top_n_features': 100,
            'importance_file': os.path.join(MODEL_DIR, 'lower_face_feature_importance.csv')
        },
        'training': {
            'test_size': 0.25,
            'random_state': 42,
            'early_stopping_rounds': 20,
            'use_ensemble': True,  # Lower face performs well without ensemble
            'model_params': {  # Default parameters if tuning is disabled
                'objective': 'multi:softprob',
                'num_class': 3,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'gamma': 0,
                'n_estimators': 100,
                'class_weights': {0: 1.0, 1: 3.5, 2: 3.0}  # Slightly higher weight for Partial
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'method': 'optuna',
                'optuna': {
                    'n_trials': 100,
                    'cv_folds': 5,
                    'direction': 'maximize',
                    'scoring': 'composite',  # Enhanced composite scoring
                    'sampler': 'TPESampler',
                    'pruner': 'MedianPruner',
                    'param_distributions': {
                        'learning_rate': ['float', 0.001, 0.3, {'log': True}],
                        'max_depth': ['int', 3, 10],
                        'n_estimators': ['int', 100, 800],
                        'min_child_weight': ['int', 1, 10],
                        'gamma': ['float', 0.0, 0.5],
                        'subsample': ['float', 0.5, 1.0],
                        'colsample_bytree': ['float', 0.5, 1.0],
                        'reg_alpha': ['float', 0.0, 0.5],
                        'reg_lambda': ['float', 0.0, 1.0],
                    },
                    'optuna_early_stopping_rounds': 20
                }
            },
            'smote': {
                'enabled': True,
                'variant': 'regular',  # Works well for lower face
                'k_neighbors': 5,
                'sampling_strategy': 'auto',
                'apply_per_fold_in_tuning': True,
                'min_samples_per_class': 50
            },
            'review_analysis': {
                'enabled': True,
                'top_k_influence': 50,
                'entropy_quantile': 0.9,
                'margin_quantile': 0.1,
                'true_label_prob_threshold': 0.4
            }
        }
    },

    'mid': {
        'name': 'Mid Face',
        'actions': ['ES', 'ET', 'BK'],
        'aus': ['AU45_r', 'AU07_r', 'AU06_r'],
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
            'critical_errors_report': os.path.join(ANALYSIS_DIR, 'midface_critical_errors.txt'),
            'partial_errors_report': os.path.join(ANALYSIS_DIR, 'midface_partial_errors.txt'),
            'review_candidates_csv': os.path.join(ANALYSIS_DIR, 'midface_review_candidates.csv')
        },
        'feature_extraction': {
            'use_normalized': True,
            'percent_diff_cap': 200.0,
            'min_value': 0.0001
        },
        'feature_selection': {
            'enabled': True,
            'top_n_features': 30,  # Increased significantly from 5
            'importance_file': os.path.join(MODEL_DIR, 'mid_face_feature_importance.csv')
        },
        'training': {
            'test_size': 0.25,
            'random_state': 42,
            'early_stopping_rounds': 25,
            'use_ensemble': True,  # Enable ensemble for problematic zone
            'model_params': {
                'objective': 'multi:softprob',
                'num_class': 3,
                'learning_rate': 0.05,
                'max_depth': 4,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'n_estimators': 200,
                'class_weights': {0: 1.0, 1: 6.0, 2: 6.0}  # Aggressive weights for minority classes
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'method': 'optuna',
                'optuna': {
                    'n_trials': 150,  # More trials for difficult zone
                    'cv_folds': 5,
                    'direction': 'maximize',
                    'scoring': 'composite',  # Use composite scoring to avoid single-class predictions
                    'sampler': 'TPESampler',
                    'pruner': 'HyperbandPruner',  # More aggressive pruning
                    'param_distributions': {
                        'learning_rate': ['float', 0.001, 0.5, {'log': True}],
                        'max_depth': ['int', 2, 12],
                        'n_estimators': ['int', 200, 1200],
                        'min_child_weight': ['int', 1, 15],
                        'gamma': ['float', 0.0, 1.0],
                        'subsample': ['float', 0.3, 1.0],
                        'colsample_bytree': ['float', 0.3, 1.0],
                        'reg_alpha': ['float', 0.0, 1.0],
                        'reg_lambda': ['float', 0.0, 2.0],
                        'scale_pos_weight': ['float', 1.0, 10.0],  # Handle imbalance
                    },
                    'optuna_early_stopping_rounds': 30
                }
            },
            'smote': {
                'enabled': True,
                'variant': 'borderline',  # Better for difficult boundaries
                'k_neighbors': 3,  # Reduced due to small minority classes
                'sampling_strategy': {  # Custom strategy to ensure minimum samples
                    0: 'auto',  # Keep majority as is
                    1: 100,  # Ensure at least 100 Partial samples
                    2: 100  # Ensure at least 100 Complete samples
                },
                'apply_per_fold_in_tuning': True,
                'min_samples_per_class': 80
            },
            'review_analysis': {
                'enabled': True,
                'top_k_influence': 75,  # More candidates for problematic zone
                'entropy_quantile': 0.85,
                'margin_quantile': 0.15,
                'true_label_prob_threshold': 0.35
            }
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
            'critical_errors_report': os.path.join(ANALYSIS_DIR, 'upperface_critical_errors.txt'),
            'partial_errors_report': os.path.join(ANALYSIS_DIR, 'upperface_partial_errors.txt'),
            'review_candidates_csv': os.path.join(ANALYSIS_DIR, 'upperface_review_candidates.csv')
        },
        'feature_extraction': {
            'use_normalized': True,
            'percent_diff_cap': 200.0,
            'min_value': 0.0001
        },
        'feature_selection': {
            'enabled': True,
            'top_n_features': 12,  # Slightly increased from 9
            'importance_file': os.path.join(MODEL_DIR, 'upper_face_feature_importance.csv')
        },
        'training': {
            'test_size': 0.25,
            'random_state': 42,
            'early_stopping_rounds': 20,
            'use_ensemble': True,  # Upper face already performs well
            'model_params': {
                'objective': 'multi:softprob',
                'num_class': 3,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'gamma': 0,
                'n_estimators': 100,
                'class_weights': {0: 1.0, 1: 4.0, 2: 2.5}  # Partial class needs more weight
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'method': 'optuna',
                'optuna': {
                    'n_trials': 100,
                    'cv_folds': 5,
                    'direction': 'maximize',
                    'scoring': 'f1_weighted',  # Upper face can use standard scoring
                    'sampler': 'TPESampler',
                    'pruner': 'MedianPruner',
                    'param_distributions': {
                        'learning_rate': ['float', 0.01, 0.3, {'log': True}],
                        'max_depth': ['int', 3, 8],
                        'n_estimators': ['int', 100, 600],
                        'min_child_weight': ['int', 1, 8],
                        'gamma': ['float', 0.0, 0.5],
                        'subsample': ['float', 0.6, 1.0],
                        'colsample_bytree': ['float', 0.6, 1.0],
                        'reg_alpha': ['float', 0.0, 0.5],
                        'reg_lambda': ['float', 0.0, 1.0],
                    },
                    'optuna_early_stopping_rounds': 20
                }
            },
            'smote': {
                'enabled': True,
                'variant': 'adasyn',  # Adaptive synthetic sampling
                'k_neighbors': 5,
                'sampling_strategy': 'auto',
                'apply_per_fold_in_tuning': True,
                'min_samples_per_class': 40
            },
            'review_analysis': {
                'enabled': True,
                'top_k_influence': 50,
                'entropy_quantile': 0.9,
                'margin_quantile': 0.1,
                'true_label_prob_threshold': 0.4
            }
        }
    }
}

# --- Review Configuration ---
REVIEW_CONFIG = {
    'similarity_threshold': 0.95,
    'consistency_checks': {
        'cross_zone': True,
        'temporal': True,
        'feature_based': True
    },
    'priority_weights': {
        'confidence': 0.3,
        'error_severity': 0.3,
        'inconsistency': 0.2,
        'influence': 0.2
    },
    'export_format': 'xlsx',
    'include_features': True,
    'max_similar_patients': 5,
    'validation': {
        'quick_validation_folds': 3,
        'min_improvement_threshold': 0.01,
        'significance_level': 0.05
    },
    'change_limits': {
        'max_changes_per_tier': {
            1: 30,
            2: 50,
            3: 100,
            4: 200
        },
        'max_distribution_shift': 0.05
    },
    'review_tiers': {
        1: {
            'name': 'Critical/High Confidence Errors',
            'description': 'Critical errors (None<->Complete) or high confidence misclassifications',
            'priority': 'highest'
        },
        2: {
            'name': 'Consistency Issues',
            'description': 'Patients with similar features but different labels',
            'priority': 'high'
        },
        3: {
            'name': 'High Uncertainty',
            'description': 'Cases with high model uncertainty',
            'priority': 'medium'
        },
        4: {
            'name': 'General Review',
            'description': 'Other cases flagged for review',
            'priority': 'low'
        }
    }
}

# --- Advanced Training Options ---
ADVANCED_TRAINING_CONFIG = {
    'cross_validation': {
        'enabled': True,
        'folds': 5,
        'shuffle': True,
        'stratified': True
    },
    'ensemble_options': {
        'voting_type': 'soft',
        'weights': {
            'xgboost': 0.7,
            'random_forest': 0.3
        },
        'random_forest_params': {
            'n_estimators': 300,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced_subsample'
        }
    },
    'calibration': {
        'method': 'isotonic',  # or 'sigmoid'
        'cv': 'prefit'
    },
    'evaluation_metrics': [
        'accuracy',
        'balanced_accuracy',
        'f1_weighted',
        'f1_macro',
        'f1_per_class',
        'cohen_kappa',
        'auc_per_class',
        'confusion_matrix',
        'classification_report'
    ],
    'monitoring': {
        'save_intermediate_results': True,
        'plot_optimization_history': True,
        'calculate_feature_importance': True,
        'generate_learning_curves': False
    }
}

# --- Data Augmentation Options (if needed) ---
DATA_AUGMENTATION_CONFIG = {
    'enabled': False,
    'methods': {
        'noise_injection': {
            'enabled': False,
            'noise_level': 0.01
        },
        'feature_perturbation': {
            'enabled': False,
            'perturbation_factor': 0.05
        }
    }
}


# --- Export Functions ---
def get_zone_config(zone):
    """Get configuration for a specific zone"""
    return ZONE_CONFIG.get(zone, {})


def get_all_zones():
    """Get list of all configured zones"""
    return list(ZONE_CONFIG.keys())


def get_model_path(zone):
    """Get model file path for a zone"""
    zone_config = get_zone_config(zone)
    return zone_config.get('filenames', {}).get('model', None)


def get_training_params(zone):
    """Get training parameters for a zone"""
    zone_config = get_zone_config(zone)
    return zone_config.get('training', {})


def update_zone_config(zone, updates):
    """Update configuration for a specific zone"""
    if zone in ZONE_CONFIG:
        # Deep update logic here
        pass


# --- Validation ---
def validate_config():
    """Validate configuration consistency"""
    errors = []

    for zone, config in ZONE_CONFIG.items():
        # Check required fields
        if 'name' not in config:
            errors.append(f"Zone {zone}: missing 'name' field")

        if 'training' not in config:
            errors.append(f"Zone {zone}: missing 'training' field")

        # Check file paths
        filenames = config.get('filenames', {})
        for key, path in filenames.items():
            if not path:
                errors.append(f"Zone {zone}: empty path for {key}")

    return errors


# Run validation on import
validation_errors = validate_config()
if validation_errors:
    print("Configuration validation errors:")
    for error in validation_errors:
        print(f"  - {error}")