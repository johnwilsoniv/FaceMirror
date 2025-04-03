"""
Facial paralysis analysis module.
Analyzes lower face paralysis detection results.
"""

import numpy as np
import logging
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

def analyze_paralysis_results(results, patient_id=None):
    """
    Analyze facial paralysis detection results.
    
    Args:
        results (dict): Results dictionary with paralysis detection data
        patient_id (str, optional): Patient ID for logging
        
    Returns:
        dict: Analysis results with metrics and insights
    """
    if not results:
        logger.warning("No results to analyze")
        return {}
    
    analysis = {
        'patient_id': patient_id,
        'paralysis_detected': False,
        'zones': defaultdict(dict),
        'affected_sides': [],
        'confidence_scores': {},
        'partial_paralysis_zones': [],
        'complete_paralysis_zones': [],
        'suggestions': []
    }
    
    # Check if paralysis was detected in any zone
    for action, info in results.items():
        if 'paralysis' not in info:
            continue
            
        if info['paralysis'].get('detected', False):
            analysis['paralysis_detected'] = True
            
        # Analyze zones
        for side in ['left', 'right']:
            for zone in ['upper', 'mid', 'lower']:
                if (zone == 'lower' and 
                    'zones' in info['paralysis'] and 
                    side in info['paralysis']['zones'] and 
                    zone in info['paralysis']['zones'][side]):
                    
                    result = info['paralysis']['zones'][side][zone]
                    
                    # Track zone results
                    if result != 'None':
                        zone_key = f"{side}_{zone}"
                        analysis['zones'][zone_key]['result'] = result
                        
                        # Track confidence
                        if ('confidence' in info['paralysis'] and 
                            side in info['paralysis']['confidence'] and 
                            zone in info['paralysis']['confidence'][side]):
                            confidence = info['paralysis']['confidence'][side][zone]
                            analysis['zones'][zone_key]['confidence'] = confidence
                            analysis['confidence_scores'][zone_key] = confidence
                        
                        # Track affected sides
                        if side not in analysis['affected_sides']:
                            analysis['affected_sides'].append(side)
                            
                        # Track by severity
                        if result == 'Partial':
                            analysis['partial_paralysis_zones'].append(zone_key)
                        elif result == 'Complete':
                            analysis['complete_paralysis_zones'].append(zone_key)
    
    # Get model details for lower face
    for action, info in results.items():
        if ('paralysis' in info and 
            'ml_details' in info['paralysis']):
            
            for key, details in info['paralysis']['ml_details'].items():
                if 'lower' in key:
                    side = key.split('_')[0]
                    zone_key = f"{side}_lower"
                    
                    if zone_key in analysis['zones']:
                        analysis['zones'][zone_key]['probabilities'] = details.get('prediction_proba', [])
                        analysis['zones'][zone_key]['confidence'] = details.get('confidence', 0)
    
    # Add analysis suggestions
    if analysis['paralysis_detected']:
        if len(analysis['partial_paralysis_zones']) > 0:
            analysis['suggestions'].append("Partial paralysis detected. Consider follow-up evaluation.")
        
        if len(analysis['complete_paralysis_zones']) > 0:
            analysis['suggestions'].append("Complete paralysis detected. Intervention may be needed.")
            
        # Add low confidence suggestion
        low_confidence_zones = [zone for zone, conf in analysis['confidence_scores'].items() 
                               if conf < 0.6]
        if low_confidence_zones:
            analysis['suggestions'].append(
                f"Detection confidence is low for {', '.join(low_confidence_zones)}. "
                f"Consider clinical verification."
            )
    
    return analysis