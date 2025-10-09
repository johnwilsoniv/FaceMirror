# synkinesis_config.py (v4.3 - Brow Cocked Ensemble Enabled for Test & Calibration Fix)

import os
from scipy.stats import uniform, randint  # Needed for Optuna if used

# --- Base Paths ---
MODEL_PARENT_DIR = 'models/synkinesis'
LOG_DIR = 'logs'
ANALYSIS_DIR = 'analysis_results'

# --- Ensure Directories Exist ---
os.makedirs(MODEL_PARENT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# --- Global Settings ---
CLASS_NAMES = {0: 'None', 1: 'Synkinesis'}
LOGGING_CONFIG = {
    'level': 'INFO',  # Kept DEBUG for thorough logging during fixes
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
}
INPUT_FILES = {
    'results_csv': 'combined_results.csv',
    'expert_key_csv': 'FPRS FP Key.csv'
}

# --- Common Optuna Parameter Distributions ---
OPTUNA_PARAM_DIST_BINARY = {
    'learning_rate': ['float', 0.005, 0.2, {'log': True}],
    'max_depth': ['int', 3, 8],
    'n_estimators': ['int', 100, 600],
    'min_child_weight': ['int', 1, 10],
    'gamma': ['float', 0.0, 0.5],
    'subsample': ['float', 0.5, 1.0],
    'colsample_bytree': ['float', 0.5, 1.0],
    'reg_alpha': ['float', 0.0, 0.5],
    'reg_lambda': ['float', 0.0, 1.0],
    'scale_pos_weight': ['float', 1.0, 20.0]  # Broad range for moderately imbalanced cases
}

# Specific Optuna params for potentially more difficult/imbalanced types
OPTUNA_PARAM_DIST_BINARY_CONSTRAINED = {
    'learning_rate': ['float', 0.005, 0.1, {'log': True}],
    'max_depth': ['int', 3, 5],
    'n_estimators': ['int', 100, 400],
    'min_child_weight': ['int', 3, 10],
    'gamma': ['float', 0.05, 0.5],
    'subsample': ['float', 0.5, 0.8],
    'colsample_bytree': ['float', 0.5, 0.8],
    'reg_alpha': ['float', 0.05, 0.8],
    'reg_lambda': ['float', 0.05, 1.0],
    'scale_pos_weight': ['float', 5.0, 25.0]  # Expanded range for highly imbalanced cases
}

# --- Synkinesis Type Specific Configurations ---
SYNKINESIS_CONFIG = {
    # --- Ocular-Oral ---
    'ocular_oral': {
        'name': 'Ocular-Oral Synkinesis',
        'relevant_actions': ['ET', 'ES', 'RE', 'BK'],
        'trigger_aus': ['AU01_r', 'AU02_r', 'AU45_r'],
        'coupled_aus': ['AU12_r', 'AU25_r', 'AU14_r'],
        'context_aus': [],
        'expert_columns': {
            'left': 'Ocular-Oral Synkinesis Left',
            'right': 'Ocular-Oral Synkinesis Right'
        },
        'target_columns': {
            'left': 'Target_Left_Ocular_Oral',
            'right': 'Target_Right_Ocular_Oral'
        },
        'filenames': {
            'model': os.path.join(MODEL_PARENT_DIR, 'ocular_oral', 'model.pkl'),
            'scaler': os.path.join(MODEL_PARENT_DIR, 'ocular_oral', 'scaler.pkl'),
            'feature_list': os.path.join(MODEL_PARENT_DIR, 'ocular_oral', 'features.list'),
            'importance': os.path.join(MODEL_PARENT_DIR, 'ocular_oral', 'feature_importance.csv'),
            'training_log': os.path.join(LOG_DIR, 'ocular_oral_training.log'),
            'analysis_log': os.path.join(ANALYSIS_DIR, 'ocular_oral', 'ocular_oral_analysis.log'),
            'error_report': os.path.join(ANALYSIS_DIR, 'ocular_oral', 'ocular_oral_errors.txt'),
            'threshold_eval_csv': os.path.join(ANALYSIS_DIR, 'ocular_oral', 'ocular_oral_threshold_evaluation.csv'),
            'pr_curve_png': os.path.join(ANALYSIS_DIR, 'ocular_oral', 'ocular_oral_precision_recall_curve.png'),
            'review_candidates_csv': os.path.join(ANALYSIS_DIR, 'ocular_oral', 'ocular_oral_review_candidates.csv'),
            'error_details_csv': os.path.join(ANALYSIS_DIR, 'ocular_oral', 'ocular_oral_error_details.csv')
        },
        'feature_extraction': {
            'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0
        },
        'feature_selection': {
            'enabled': True,
            'top_n_features': 40,
            'importance_file': os.path.join(MODEL_PARENT_DIR, 'ocular_oral', 'feature_importance.csv')
        },
        'training': {
            'test_size': 0.25, 'random_state': 42,
            'early_stopping_rounds': 20,
            'use_ensemble': True,
            'model_params': {
                'objective': 'binary:logistic', 'eval_metric': 'aucpr',
                'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0,
                'n_estimators': 150, 'scale_pos_weight': 4.0,
                'random_state': 42
            },
            'hyperparameter_tuning': {
                'enabled': True, 'method': 'optuna',
                'optuna': {
                    'n_trials': 75,
                    'cv_folds': 5, 'direction': 'maximize', 'scoring': 'average_precision',
                    'sampler': 'TPESampler', 'pruner': 'MedianPruner',
                    'param_distributions': OPTUNA_PARAM_DIST_BINARY_CONSTRAINED,
                    'optuna_early_stopping_rounds': 15
                }
            },
            'smote': {
                'enabled': True, 'variant': 'adasyn', 'k_neighbors': 3,
                'sampling_strategy': {1: 70},
                'apply_per_fold_in_tuning': True,
                'apply_to_full_train_if_not_per_fold': True,
                'min_samples_per_class': 5
            },
            'calibration': {  # Per-type calibration settings
                'method': 'sigmoid',
                'cv': 'prefit',
                'calibration_split_size': 0.2,
                'min_samples_per_class_prefit': 10  # Specific min samples for this type's prefit calib
            },
            'review_analysis': {
                'enabled': True, 'top_k_influence': 30, 'entropy_quantile': 0.9,
                'margin_quantile': 0.1, 'true_label_prob_threshold': 0.4
            }
        }
    },
    # --- Oral-Ocular ---
    'oral_ocular': {
        'name': 'Oral-Ocular Synkinesis',
        'relevant_actions': ['BS', 'SS', 'SO', 'SE', 'PL', 'LT'],
        'trigger_aus': ['AU12_r', 'AU25_r'],
        'coupled_aus': ['AU06_r', 'AU45_r'],
        'context_aus': [],
        'expert_columns': {'left': 'Oral-Ocular Synkinesis Left', 'right': 'Oral-Ocular Synkinesis Right'},
        'target_columns': {'left': 'Target_Left_Oral_Ocular', 'right': 'Target_Right_Oral_Ocular'},
        'filenames': {
            'model': os.path.join(MODEL_PARENT_DIR, 'oral_ocular', 'model.pkl'),
            'scaler': os.path.join(MODEL_PARENT_DIR, 'oral_ocular', 'scaler.pkl'),
            'feature_list': os.path.join(MODEL_PARENT_DIR, 'oral_ocular', 'features.list'),
            'importance': os.path.join(MODEL_PARENT_DIR, 'oral_ocular', 'feature_importance.csv'),
            'training_log': os.path.join(LOG_DIR, 'oral_ocular_training.log'),
            'analysis_log': os.path.join(ANALYSIS_DIR, 'oral_ocular', 'oral_ocular_analysis.log'),
            'error_report': os.path.join(ANALYSIS_DIR, 'oral_ocular', 'oral_ocular_errors.txt'),
            'threshold_eval_csv': os.path.join(ANALYSIS_DIR, 'oral_ocular', 'oral_ocular_threshold_evaluation.csv'),
            'pr_curve_png': os.path.join(ANALYSIS_DIR, 'oral_ocular', 'oral_ocular_precision_recall_curve.png'),
            'review_candidates_csv': os.path.join(ANALYSIS_DIR, 'oral_ocular', 'oral_ocular_review_candidates.csv'),
            'error_details_csv': os.path.join(ANALYSIS_DIR, 'oral_ocular', 'oral_ocular_error_details.csv')
        },
        'feature_extraction': {'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0},
        'feature_selection': {
            'enabled': True, 'top_n_features': 40,
            'importance_file': os.path.join(MODEL_PARENT_DIR, 'oral_ocular', 'feature_importance.csv')
        },
        'training': {
            'test_size': 0.25, 'random_state': 42,
            'early_stopping_rounds': 20,
            'use_ensemble': True,
            'model_params': {
                'objective': 'binary:logistic', 'eval_metric': 'aucpr',
                'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0,
                'n_estimators': 150, 'scale_pos_weight': 2.0,
                'random_state': 42
            },
            'hyperparameter_tuning': {
                'enabled': True, 'method': 'optuna',
                'optuna': {
                    'n_trials': 50, 'cv_folds': 5, 'direction': 'maximize', 'scoring': 'average_precision',
                    'sampler': 'TPESampler', 'pruner': 'MedianPruner',
                    'param_distributions': OPTUNA_PARAM_DIST_BINARY,
                    'optuna_early_stopping_rounds': 15
                }
            },
            'smote': {
                'enabled': True, 'variant': 'borderline', 'k_neighbors': 5,
                'sampling_strategy': 'auto',
                'apply_per_fold_in_tuning': True,
                'apply_to_full_train_if_not_per_fold': True,
                'min_samples_per_class': 30
            },
            'calibration': {
                'method': 'sigmoid', 'cv': 'prefit', 'calibration_split_size': 0.2,
                'min_samples_per_class_prefit': 10
            },
            'review_analysis': {
                'enabled': True, 'top_k_influence': 30, 'entropy_quantile': 0.9,
                'margin_quantile': 0.1, 'true_label_prob_threshold': 0.4
            }
        }
    },
    # --- Snarl-Smile ---
    'snarl_smile': {
        'name': 'Snarl-Smile Synkinesis',
        'relevant_actions': ['BS'],
        'trigger_aus': ['AU12_r'],
        'coupled_aus': ['AU10_r', 'AU14_r', 'AU15_r'],
        'context_aus': [],
        'expert_columns': {'left': 'Snarl Smile Left', 'right': 'Snarl Smile Right'},
        'target_columns': {'left': 'Target_Left_Snarl_Smile', 'right': 'Target_Right_Snarl_Smile'},
        'filenames': {
            'model': os.path.join(MODEL_PARENT_DIR, 'snarl_smile', 'model.pkl'),
            'scaler': os.path.join(MODEL_PARENT_DIR, 'snarl_smile', 'scaler.pkl'),
            'feature_list': os.path.join(MODEL_PARENT_DIR, 'snarl_smile', 'features.list'),
            'importance': os.path.join(MODEL_PARENT_DIR, 'snarl_smile', 'feature_importance.csv'),
            'training_log': os.path.join(LOG_DIR, 'snarl_smile_training.log'),
            'analysis_log': os.path.join(ANALYSIS_DIR, 'snarl_smile', 'snarl_smile_analysis.log'),
            'error_report': os.path.join(ANALYSIS_DIR, 'snarl_smile', 'snarl_smile_errors.txt'),
            'threshold_eval_csv': os.path.join(ANALYSIS_DIR, 'snarl_smile', 'snarl_smile_threshold_evaluation.csv'),
            'pr_curve_png': os.path.join(ANALYSIS_DIR, 'snarl_smile', 'snarl_smile_precision_recall_curve.png'),
            'review_candidates_csv': os.path.join(ANALYSIS_DIR, 'snarl_smile', 'snarl_smile_review_candidates.csv'),
            'error_details_csv': os.path.join(ANALYSIS_DIR, 'snarl_smile', 'snarl_smile_error_details.csv')
        },
        'feature_extraction': {'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0},
        'feature_selection': {
            'enabled': True, 'top_n_features': 15,
            'importance_file': os.path.join(MODEL_PARENT_DIR, 'snarl_smile', 'feature_importance.csv')
        },
        'training': {
            'test_size': 0.25, 'random_state': 42,
            'early_stopping_rounds': 15,
            'use_ensemble': True,
            'model_params': {
                'objective': 'binary:logistic', 'eval_metric': 'aucpr',
                'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 1,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.5,
                'reg_alpha': 0.5, 'n_estimators': 100, 'scale_pos_weight': 2.5,
                'random_state': 42
            },
            'hyperparameter_tuning': {
                'enabled': True, 'method': 'optuna',
                'optuna': {
                    'n_trials': 50, 'cv_folds': 5, 'direction': 'maximize', 'scoring': 'average_precision',
                    'sampler': 'TPESampler', 'pruner': 'MedianPruner',
                    'param_distributions': OPTUNA_PARAM_DIST_BINARY,
                    'optuna_early_stopping_rounds': 15
                }
            },
            'smote': {
                'enabled': True, 'variant': 'borderline', 'k_neighbors': 5,
                'sampling_strategy': 'auto',
                'apply_per_fold_in_tuning': True,
                'apply_to_full_train_if_not_per_fold': True,
                'min_samples_per_class': 30
            },
            'calibration': {
                'method': 'sigmoid', 'cv': 'prefit', 'calibration_split_size': 0.2,
                'min_samples_per_class_prefit': 10
            },
            'review_analysis': {
                'enabled': True, 'top_k_influence': 30, 'entropy_quantile': 0.9,
                'margin_quantile': 0.1, 'true_label_prob_threshold': 0.4
            }
        }
    },
    # --- Mentalis ---
    'mentalis': {
        'name': 'Mentalis Synkinesis',
        'relevant_actions': ['ET', 'ES', 'BS', 'SS', 'SO', 'SE', 'RE', 'PL', 'FR', 'BK', 'WN', 'BC', 'LT'],
        'trigger_aus': [],
        'coupled_aus': ['AU17_r'],
        'context_aus': ['AU12_r', 'AU15_r', 'AU16_r'],
        'expert_columns': {'left': 'Mentalis Synkinesis Left', 'right': 'Mentalis Synkinesis Right'},
        'target_columns': {'left': 'Target_Left_Mentalis', 'right': 'Target_Right_Mentalis'},
        'filenames': {
            'model': os.path.join(MODEL_PARENT_DIR, 'mentalis', 'model.pkl'),
            'scaler': os.path.join(MODEL_PARENT_DIR, 'mentalis', 'scaler.pkl'),
            'feature_list': os.path.join(MODEL_PARENT_DIR, 'mentalis', 'features.list'),
            'importance': os.path.join(MODEL_PARENT_DIR, 'mentalis', 'feature_importance.csv'),
            'training_log': os.path.join(LOG_DIR, 'mentalis_training.log'),
            'analysis_log': os.path.join(ANALYSIS_DIR, 'mentalis', 'mentalis_analysis.log'),
            'error_report': os.path.join(ANALYSIS_DIR, 'mentalis', 'mentalis_errors.txt'),
            'threshold_eval_csv': os.path.join(ANALYSIS_DIR, 'mentalis', 'mentalis_threshold_evaluation.csv'),
            'pr_curve_png': os.path.join(ANALYSIS_DIR, 'mentalis', 'mentalis_precision_recall_curve.png'),
            'review_candidates_csv': os.path.join(ANALYSIS_DIR, 'mentalis', 'mentalis_review_candidates.csv'),
            'error_details_csv': os.path.join(ANALYSIS_DIR, 'mentalis', 'mentalis_error_details.csv')
        },
        'feature_extraction': {'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0},
        'feature_selection': {
            'enabled': True, 'top_n_features': 15,
            'importance_file': os.path.join(MODEL_PARENT_DIR, 'mentalis', 'feature_importance.csv')
        },
        'training': {
            'test_size': 0.25, 'random_state': 42,
            'early_stopping_rounds': 15,
            'use_ensemble': True,
            'model_params': {
                'objective': 'binary:logistic', 'eval_metric': 'aucpr',
                'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 1,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1,
                'n_estimators': 100, 'scale_pos_weight': 2.0,
                'random_state': 42
            },
            'hyperparameter_tuning': {
                'enabled': True, 'method': 'optuna',
                'optuna': {
                    'n_trials': 50, 'cv_folds': 5, 'direction': 'maximize', 'scoring': 'average_precision',
                    'sampler': 'TPESampler', 'pruner': 'MedianPruner',
                    'param_distributions': OPTUNA_PARAM_DIST_BINARY,
                    'optuna_early_stopping_rounds': 15
                }
            },
            'smote': {
                'enabled': True, 'variant': 'borderline', 'k_neighbors': 5,
                'sampling_strategy': 'auto',
                'apply_per_fold_in_tuning': True,
                'apply_to_full_train_if_not_per_fold': True,
                'min_samples_per_class': 30
            },
            'calibration': {
                'method': 'sigmoid', 'cv': 'prefit', 'calibration_split_size': 0.2,
                'min_samples_per_class_prefit': 10
            },
            'review_analysis': {
                'enabled': True, 'top_k_influence': 30, 'entropy_quantile': 0.9,
                'margin_quantile': 0.1, 'true_label_prob_threshold': 0.4
            }
        }
    },
    # --- Hypertonicity ---
    'hypertonicity': {
        'name': 'Hypertonicity',
        'relevant_actions': ['BL'],
        'trigger_aus': [], 'coupled_aus': [], 'context_aus': [],
        'interest_aus': ['AU12_r', 'AU14_r'],
        'expert_columns': {'left': 'Hypertonicity Left', 'right': 'Hypertonicity Right'},
        'target_columns': {'left': 'Target_Left_Hypertonicity', 'right': 'Target_Right_Hypertonicity'},
        'filenames': {
            'model': os.path.join(MODEL_PARENT_DIR, 'hypertonicity', 'model.pkl'),
            'scaler': os.path.join(MODEL_PARENT_DIR, 'hypertonicity', 'scaler.pkl'),
            'feature_list': os.path.join(MODEL_PARENT_DIR, 'hypertonicity', 'features.list'),
            'importance': os.path.join(MODEL_PARENT_DIR, 'hypertonicity', 'feature_importance.csv'),
            'training_log': os.path.join(LOG_DIR, 'hypertonicity_training.log'),
            'analysis_log': os.path.join(ANALYSIS_DIR, 'hypertonicity', 'hypertonicity_analysis.log'),
            'error_report': os.path.join(ANALYSIS_DIR, 'hypertonicity', 'hypertonicity_errors.txt'),
            'threshold_eval_csv': os.path.join(ANALYSIS_DIR, 'hypertonicity', 'hypertonicity_threshold_evaluation.csv'),
            'pr_curve_png': os.path.join(ANALYSIS_DIR, 'hypertonicity', 'hypertonicity_precision_recall_curve.png'),
            'review_candidates_csv': os.path.join(ANALYSIS_DIR, 'hypertonicity', 'hypertonicity_review_candidates.csv'),
            'error_details_csv': os.path.join(ANALYSIS_DIR, 'hypertonicity', 'hypertonicity_error_details.csv')
        },
        'feature_extraction': {'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0},
        'feature_selection': {
            'enabled': True, 'top_n_features': 20,
            'importance_file': os.path.join(MODEL_PARENT_DIR, 'hypertonicity', 'feature_importance.csv')
        },
        'training': {
            'test_size': 0.25, 'random_state': 42,
            'early_stopping_rounds': 15,
            'use_ensemble': True,
            'model_params': {
                'objective': 'binary:logistic', 'eval_metric': 'aucpr',
                'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 1,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1,
                'n_estimators': 100, 'scale_pos_weight': 1.5,
                'random_state': 42
            },
            'hyperparameter_tuning': {
                'enabled': True, 'method': 'optuna',
                'optuna': {
                    'n_trials': 50, 'cv_folds': 5, 'direction': 'maximize', 'scoring': 'average_precision',
                    'sampler': 'TPESampler', 'pruner': 'MedianPruner',
                    'param_distributions': OPTUNA_PARAM_DIST_BINARY,
                    'optuna_early_stopping_rounds': 15
                }
            },
            'smote': {
                'enabled': True, 'variant': 'regular', 'k_neighbors': 5,
                'sampling_strategy': 'auto',
                'apply_per_fold_in_tuning': True,
                'apply_to_full_train_if_not_per_fold': True,
                'min_samples_per_class': 30
            },
            'calibration': {
                'method': 'sigmoid', 'cv': 'prefit', 'calibration_split_size': 0.2,
                'min_samples_per_class_prefit': 10
            },
            'review_analysis': {
                'enabled': True, 'top_k_influence': 30, 'entropy_quantile': 0.9,
                'margin_quantile': 0.1, 'true_label_prob_threshold': 0.4
            }
        }
    },
    # --- Brow Cocked --- MODIFIED for V3 Features and Ensemble Test ---
    'brow_cocked': {
        'name': 'Brow Cocked Synkinesis',
        'relevant_actions': ['BL', 'RE'],  # V3: Focus for features is BL and RE
        'trigger_aus': [],  # V3: Not an ET-triggered synkinesis by this definition
        'coupled_aus': ['AU01_r', 'AU02_r'],  # These are the AUs of interest for brow elevation
        'interest_aus': ['AU01_r', 'AU02_r'],  # Brow AUs
        'context_aus': [],  # V3: AU07 (lid tightener) no longer primary context for features
        'expert_columns': {'left': 'Brow Cocked Left', 'right': 'Brow Cocked Right'},
        'target_columns': {'left': 'Target_Left_BrowCocked', 'right': 'Target_Right_BrowCocked'},
        'filenames': {
            'model': os.path.join(MODEL_PARENT_DIR, 'brow_cocked', 'model.pkl'),
            'scaler': os.path.join(MODEL_PARENT_DIR, 'brow_cocked', 'scaler.pkl'),
            'feature_list': os.path.join(MODEL_PARENT_DIR, 'brow_cocked', 'features.list'),
            'importance': os.path.join(MODEL_PARENT_DIR, 'brow_cocked', 'feature_importance.csv'),
            'training_log': os.path.join(LOG_DIR, 'brow_cocked_training.log'),
            'analysis_log': os.path.join(ANALYSIS_DIR, 'brow_cocked', 'brow_cocked_analysis.log'),
            'error_report': os.path.join(ANALYSIS_DIR, 'brow_cocked', 'brow_cocked_errors.txt'),
            'threshold_eval_csv': os.path.join(ANALYSIS_DIR, 'brow_cocked', 'brow_cocked_threshold_evaluation.csv'),
            'pr_curve_png': os.path.join(ANALYSIS_DIR, 'brow_cocked', 'brow_cocked_precision_recall_curve.png'),
            'review_candidates_csv': os.path.join(ANALYSIS_DIR, 'brow_cocked', 'brow_cocked_review_candidates.csv'),
            'error_details_csv': os.path.join(ANALYSIS_DIR, 'brow_cocked', 'brow_cocked_error_details.csv')
        },
        'feature_extraction': {'use_normalized': True, 'min_value': 0.0001, 'percent_diff_cap': 200.0},
        'feature_selection': {  # Features for V3 definition might be different, may need re-evaluation later
            'enabled': True, 'top_n_features': 10,  # This N might need adjustment after V3 features
            'importance_file': os.path.join(MODEL_PARENT_DIR, 'brow_cocked', 'feature_importance.csv')
        },
        'training': {
            'test_size': 0.25, 'random_state': 42,
            'early_stopping_rounds': 15,
            'use_ensemble': True,  # MODIFIED: Testing with ensemble enabled
            'model_params': {
                'objective': 'binary:logistic', 'eval_metric': 'aucpr',
                'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 1,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1,
                'n_estimators': 100, 'scale_pos_weight': 8.0,
                # This was from Optuna, may need re-tune with new features
                'random_state': 42
            },
            'hyperparameter_tuning': {
                'enabled': False, 'method': 'optuna',
                'optuna': {
                    'n_trials': 75,  # Might need more trials with new feature set
                    'cv_folds': 5, 'direction': 'maximize', 'scoring': 'average_precision',
                    'sampler': 'TPESampler', 'pruner': 'MedianPruner',
                    'param_distributions': OPTUNA_PARAM_DIST_BINARY_CONSTRAINED,
                    'optuna_early_stopping_rounds': 15
                }
            },
            'smote': {  # Highly imbalanced, ADASYN and specific target count might be good
                'enabled': True, 'variant': 'adasyn', 'k_neighbors': 3,
                'sampling_strategy': {1: 100},  # Target a specific number for positive class
                'apply_per_fold_in_tuning': True,
                'apply_to_full_train_if_not_per_fold': True,
                'min_samples_per_class': 5  # ADASYN can work with very small classes
            },
            'calibration': {  # Per-type calibration settings
                'method': 'sigmoid',
                'cv': 'prefit',
                'calibration_split_size': 0.2,
                'min_samples_per_class_prefit': 10  # Ensure enough samples for calibration
            },
            'review_analysis': {
                'enabled': True, 'top_k_influence': 30, 'entropy_quantile': 0.9,
                'margin_quantile': 0.1, 'true_label_prob_threshold': 0.4
            }
        }
    },
}

# --- Review Configuration ---
REVIEW_CONFIG = {
    'similarity_threshold': 0.95,
    'consistency_checks': {
        'cross_synk_type': True, 'temporal': True, 'feature_based': True
    },
    'priority_weights': {
        'confidence': 0.4, 'error_severity': 0.0, 'inconsistency': 0.3, 'influence': 0.3
    },
    'export_format': 'xlsx', 'include_features': True, 'max_similar_patients': 5,
    'validation': {
        'quick_validation_folds': 3, 'min_improvement_threshold': 0.01, 'significance_level': 0.05
    },
    'change_limits': {
        'max_changes_per_tier': {1: 20, 2: 40, 3: 80, 4: 150},
        'max_distribution_shift': 0.05
    },
    'review_tiers': {
        1: {'name': 'High Confidence Errors (FP/FN)',
            'description': 'False Positives or False Negatives with high model confidence.', 'priority': 'highest'},
        2: {'name': 'Consistency Issues',
            'description': 'Patients with similar features but different synkinesis labels.', 'priority': 'high'},
        3: {'name': 'High Uncertainty (Borderline Cases)',
            'description': 'Cases where model probability is close to 0.5.', 'priority': 'medium'},
        4: {'name': 'General Review Pool',
            'description': 'Other cases flagged, possibly random sample of positive class.', 'priority': 'low'}
    }
}

# --- Advanced Training Options ---
ADVANCED_TRAINING_CONFIG = {
    'cross_validation': {
        'enabled': False, 'folds': 5, 'shuffle': True, 'stratified': True
    },
    'ensemble_options': {
        'voting_type': 'soft',
        'weights': {'xgboost': 0.7, 'random_forest': 0.3},
        'random_forest_params': {
            'n_estimators': 150, 'max_depth': None, 'min_samples_split': 2,
            'min_samples_leaf': 1, 'class_weight': 'balanced_subsample'
        }
    },
    'calibration': {  # Global default calibration settings
        'method': 'sigmoid', 'cv': 'prefit',
        'calibration_split_size': 0.2,
        'min_samples_per_class_prefit': 10  # Global default min samples for prefit
    },
    'evaluation_metrics': [
        'accuracy_score', 'balanced_accuracy_score',
        'f1_score', 'precision_score', 'recall_score',
        'roc_auc_score', 'average_precision_score',
        'brier_score_loss', 'log_loss', 'confusion_matrix'
    ],
    'monitoring': {
        'save_intermediate_results': False, 'plot_optimization_history': True,
        'calculate_feature_importance': True, 'generate_learning_curves': False
    }
}

# --- Data Augmentation Options ---
DATA_AUGMENTATION_CONFIG = {
    'enabled': False,
    'methods': {
        'noise_injection': {'enabled': False, 'noise_level': 0.01},
        'feature_perturbation': {'enabled': False, 'perturbation_factor': 0.05}
    }
}


# --- Export Functions ---
def get_synkinesis_config(synk_type):
    return SYNKINESIS_CONFIG.get(synk_type, {})


def get_all_synkinesis_types():
    return list(SYNKINESIS_CONFIG.keys())


def get_model_path(synk_type):
    type_config = get_synkinesis_config(synk_type)
    return type_config.get('filenames', {}).get('model', None)


def get_training_params(synk_type):
    type_config = get_synkinesis_config(synk_type)
    return type_config.get('training', {})


def update_synkinesis_config(synk_type, updates):
    if synk_type in SYNKINESIS_CONFIG:
        # Deep update for nested dicts like 'training' or 'model_params'
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(SYNKINESIS_CONFIG[synk_type].get(key), dict):
                # Further recurse for sub-dictionaries if necessary, e.g. training.model_params
                current_level = SYNKINESIS_CONFIG[synk_type][key]
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict) and isinstance(current_level.get(sub_key), dict):
                        current_level[sub_key].update(sub_value)
                    else:
                        current_level[sub_key] = sub_value
            else:
                SYNKINESIS_CONFIG[synk_type][key] = value


# --- Validation ---
def validate_config():
    errors = []
    for synk_type, config_item in SYNKINESIS_CONFIG.items():
        if 'name' not in config_item: errors.append(f"Synkinesis Type {synk_type}: missing 'name' field")
        if 'training' not in config_item:
            errors.append(f"Synkinesis Type {synk_type}: missing 'training' field")
        else:
            if 'model_params' not in config_item['training']: errors.append(
                f"Synkinesis Type {synk_type}: training missing 'model_params'")
            if config_item['training']['hyperparameter_tuning'].get('enabled') and \
                    'optuna' not in config_item['training']['hyperparameter_tuning']:
                errors.append(f"Synkinesis Type {synk_type}: Optuna config missing in hyperparameter_tuning")

            # Check for per-type calibration config
            if 'calibration' not in config_item['training']:
                errors.append(f"Synkinesis Type {synk_type}: training block missing 'calibration' dictionary.")
            elif not isinstance(config_item['training']['calibration'], dict):
                errors.append(f"Synkinesis Type {synk_type}: training.calibration must be a dictionary.")
            elif 'min_samples_per_class_prefit' not in config_item['training']['calibration']:
                errors.append(
                    f"Synkinesis Type {synk_type}: training.calibration missing 'min_samples_per_class_prefit'.")

        filenames_item = config_item.get('filenames', {})
        for key, path_val in filenames_item.items():
            if not path_val:
                errors.append(f"Synkinesis Type {synk_type}: empty path for filenames.{key}")
            else:
                dir_to_create = os.path.dirname(path_val)
                if dir_to_create:
                    os.makedirs(dir_to_create, exist_ok=True)

    if 'calibration' not in ADVANCED_TRAINING_CONFIG:
        errors.append("ADVANCED_TRAINING_CONFIG missing 'calibration'")
    else:
        if 'min_samples_per_class_prefit' not in ADVANCED_TRAINING_CONFIG['calibration']:
            errors.append("ADVANCED_TRAINING_CONFIG.calibration missing 'min_samples_per_class_prefit'.")

    if 'evaluation_metrics' not in ADVANCED_TRAINING_CONFIG: errors.append(
        "ADVANCED_TRAINING_CONFIG missing 'evaluation_metrics'")

    return errors


validation_errors = validate_config()
if validation_errors:
    print("Synkinesis Configuration validation errors:")
    for error in validation_errors:
        print(f"  - {error}")