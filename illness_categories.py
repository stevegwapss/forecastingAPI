"""
Illness Category Taxonomy - 8 Categories
=========================================
Centralized illness category mapping based on actual complaint patterns 
from 8,127 patient records in the Uzi Care clinic dataset.

This taxonomy replaces the old 6-category system with a more accurate 
8-category system that includes neurological_psychological and cardiovascular.

Categories:
1. respiratory - Cough, cold, flu, asthma, throat issues
2. gastrointestinal - Stomach, digestive, UTI issues
3. pain_aches - Headaches, dysmenorrhea, toothaches, body pain
4. skin_allergy - Allergies, rashes, skin conditions, eye issues
5. injury_trauma - Wounds, cuts, sprains, bites, burns
6. neurological_psychological - Dizziness, anxiety, seizures
7. cardiovascular - Blood pressure, heart issues, bleeding
8. fever_general - Fever, weakness, general/screening visits

Author: Uzi Care Development Team
Date: October 26, 2025
Version: 2.0 (8-Category System)
"""

ILLNESS_CATEGORIES = {
    'respiratory': [
        # Common terms from dataset
        'colds', 'cold', 'cough', 'flu', 'runny nose', 'sore throat',
        'asthma', 'sinusitis', 'rhinitis', 'allergic rhinitis',
        'nasal congestion', 'throat pain', 'itchy throat', 'throat itchiness',
        'difficulty breathing', 'shortness of breath', 'sob', 'dob',
        'wheezing', 'chest pain', 'pneumonia', 'bronchitis',
        'tonsillitis', 'tonsilitis', 'pharyngitis', 'urti',
        'congested nose', 'clogged nose', 'nose itchiness',
        'difficulty swallowing', 'phlegm', 'productive cough', 'dry cough',
        # Variations found in data
        'cough and colds', 'cough/colds', 'flu-like symptoms',
        'exposure to flood',  # Often leads to respiratory issues
    ],
    
    'gastrointestinal': [
        # Digestive complaints (very common in dataset)
        'abdominal pain', 'stomach', 'stomachache', 'stomach ache',
        'hyperacidity', 'gas pain', 'acid reflux', 'gerd',
        'lbm', 'diarrhea', 'loose bowel',
        'nausea', 'vomiting', 'vomitting',  # Common misspelling
        'constipation', 'bloat', 'indigestion',
        'gastritis', 'gastroenteritis', 'peptic',
        'epigastric pain', 'heartburn',
        'abdominal cramps', 'abdominal spasm',
        # Specific areas
        'llq pain', 'right upper quadrant', 'flank pain',
        # Related terms
        'food poisoning', 'uti', 'urinary tract infection', 'dysuria',
    ],
    
    'pain_aches': [
        # TOP complaint in dataset (1,756 cases of headache alone!)
        'headache', 'migraine', 'head ache',
        
        # Second most common (706 cases!)
        'dysmenorrhea', 'dysmennorhea', 'menstrual', 'cramps',
        'menstruation cramps', 'menstruational cramps', 'menstrual pain',
        
        # Third most common (596 cases!)
        'toothache', 'tooth', 'dental', 'gum pain', 'tootache',
        'wisdom tooth', 'mouth sore', 'oral',
        
        # Body pain (173 cases)
        'body pain', 'bodyache', 'body ache', 'body weakness',
        'fatigue', 'malaise', 'weak', 'feeling cold',
        
        # Specific pain locations
        'back pain', 'backache', 'backpain', 'lower back pain', 'back ache',
        'neck pain', 'neckpain', 'stiff neck',
        'shoulder pain',
        'knee pain', 'leg pain', 'foot pain', 'ankle pain',
        'arm pain', 'hand pain', 'wrist pain',
        'joint pain', 'arthritis', 'muscle pain',
        'ear pain', 'earache', 'ear ache', 'eye pain',
        'chest pain',  # When not respiratory
        
        # General pain
        'pain', 'ache', 'soreness',
    ],
    
    'skin_allergy': [
        # Allergy (377 cases - 4th most common!)
        'allergy', 'allergic', 'allergic reaction', 'allergic rxn',
        'rash', 'rashes', 'hives', 'urticaria',
        'itch', 'itchy', 'itchiness', 'irritation',
        'skin allergy', 'skin irritation',
        
        # Dermatological
        'eczema', 'dermatitis', 'skin infection',
        'boil', 'boils', 'abscess', 'pimple',
        'blister', 'blisters', 'vesicle',
        'redness', 'swelling', 'inflammation', 'inflamed',
        'insect bite', 'ant bite', 'bee sting', 'bug',
        
        # Eyes (common in dataset)
        'eye irritation', 'eye itch', 'sore eyes', 'sore eye',
        'eye allergy', 'eye pain', 'conjunctivitis',
        'dry eyes', 'red eye', 'eye redness',
        
        # Skin conditions
        'varicella', 'chickenpox', 'measles', 'herpes zoster',
        'viral exanthem', 'stomatitis', 'mumps', 'scurvy',
        
        # Infections
        'infected', 'infection', 'pus', 'discharge',
        'impacted cerumen', 'otitis', 'earwax',
    ],
    
    'injury_trauma': [
        # Open wounds (134 cases!)
        'wound', 'open wound', 'cut', 'laceration',
        'wound dressing', 'wound cleaning', 'wound pain',
        
        # Abrasions (35 cases)
        'abrasion', 'scratch', 'scrape',
        
        # Sprains (26 cases)
        'sprain', 'strain', 'twisted', 'ankle sprain',
        
        # Trauma
        'trauma', 'injury', 'fall', 'accident',
        'mva', 'motor vehicle', 'vehicular accident',
        'fracture', 'broken', 'dislocation',
        'contusion', 'bruise', 'hematoma',
        
        # Burns
        'burn', 'burned', 'burnt',
        
        # Bites
        'bite', 'dog bite', 'dogbite', 'cat bite', 'rat bite',
        'animal bite', 'cat scratched',
        
        # Punctures
        'puncture', 'punctured', 'pricked', 'needle prick',
        'stapler', 'thumb stack',
        
        # Minor injuries
        'bump', 'bumped', 'hit',
        
        # Sports/activity related
        'sports injury', 'dragon boat',
    ],
    
    'neurological_psychological': [
        # NEW CATEGORY - Very common in dataset!
        # Dizziness (89 cases + variations)
        'dizziness', 'diziness', 'dizzines', 'vertigo',
        'lightheaded', 'faint', 'fainting', 'fainted', 'syncope',
        'loss of consciousness', 'decreased loc', 'passed out',
        
        # Anxiety (7 cases + panic attacks)
        'anxiety', 'anxiety attack', 'panic', 'panic attack',
        'depression', 'stress',
        
        # Neurological
        'seizure', 'epilepsy', 'convulsion',
        'numbness', 'tingling', 'twitching', 'spasm',
        'palpitation', 'tachycardic',
        'stroke', 'paralysis',
        'migraine',  # Severe headache type
        
        # Other
        'hyperventilation', 'hyperventilate',
        'motion sickness',
    ],
    
    'cardiovascular': [
        # NEW CATEGORY - Blood pressure issues common in dataset
        'high blood pressure', 'high bp', 'hpn', 'hypertension',
        'hypertensive', 'elevated bp', 'increased bp',
        'low bp', 'hypotension',
        'bp monitoring', 'blood pressure',
        
        # Heart-related
        'chest pain', 'heart', 'cardiac', 'angina',
        'heart failure', 'palpitation',
        
        # Circulatory
        'nosebleed', 'nosebleeding', 'nose bleeding',
        'blood', 'bleeding', 'hemorrhage',
    ],
    
    'fever_general': [
        # Fever (391 cases - 6th most common!)
        'fever', 'febrile', 'pyrexia', 'temperature',
        'chills', 'feverish', 'fever-like',
        
        # General/systemic
        'malaise', 'weakness', 'fatigue', 'tired',
        'pale', 'chills', 'sweating',
        'dehydration',
        
        # Monitoring/screening
        'blood sugar', 'monitoring', 'screening',
        'post vaccination', 'vaccination', 'prophylaxis',
        'travel purposes', 'med cert', 'medical certificate',
        
        # Non-specific
        'other', 'general', 'unspecified',
    ],
}


def categorize_complaint(complaint_text):
    """
    Categorize a patient complaint into one of 8 illness categories.
    
    Uses keyword matching against the ILLNESS_CATEGORIES taxonomy.
    Categories are checked in order, returning the first match found.
    
    Priority order (implicit by dict iteration in Python 3.7+):
    1. respiratory
    2. gastrointestinal
    3. pain_aches
    4. skin_allergy
    5. injury_trauma
    6. neurological_psychological
    7. cardiovascular
    8. fever_general
    
    Args:
        complaint_text (str): Patient complaint text
        
    Returns:
        str: Category name (one of the 8 categories)
        
    Examples:
        >>> categorize_complaint("Cough and colds")
        'respiratory'
        >>> categorize_complaint("Headache")
        'pain_aches'
        >>> categorize_complaint("Dizziness")
        'neurological_psychological'
    """
    import pandas as pd
    
    if pd.isna(complaint_text) or not complaint_text:
        return 'fever_general'  # Default for empty/null complaints
    
    complaint_lower = str(complaint_text).lower()
    
    # Check each category for keyword matches
    for category, keywords in ILLNESS_CATEGORIES.items():
        if any(keyword in complaint_lower for keyword in keywords):
            return category
    
    # No match found - default to fever_general
    return 'fever_general'


def get_all_categories():
    """
    Get list of all 8 illness categories.
    
    Returns:
        list: List of category names
    """
    return list(ILLNESS_CATEGORIES.keys())


def get_category_keywords(category):
    """
    Get all keywords for a specific category.
    
    Args:
        category (str): Category name
        
    Returns:
        list: List of keywords for that category, or None if category doesn't exist
    """
    return ILLNESS_CATEGORIES.get(category)


def print_taxonomy_summary():
    """Print a summary of the taxonomy structure."""
    print("=" * 70)
    print("ILLNESS CATEGORY TAXONOMY - 8 Categories")
    print("=" * 70)
    print(f"\nTotal Categories: {len(ILLNESS_CATEGORIES)}")
    print()
    
    for i, (category, keywords) in enumerate(ILLNESS_CATEGORIES.items(), 1):
        print(f"{i}. {category:30s} {len(keywords):3d} keywords")
    
    total_keywords = sum(len(keywords) for keywords in ILLNESS_CATEGORIES.values())
    print(f"\nTotal Keywords: {total_keywords}")
    print("=" * 70)


if __name__ == "__main__":
    # Test the taxonomy
    print_taxonomy_summary()
    
    # Test some sample complaints
    print("\n" + "=" * 70)
    print("SAMPLE CATEGORIZATION TESTS")
    print("=" * 70)
    
    test_cases = [
        "Cough and colds",
        "Headache",
        "Dizziness",
        "High blood pressure",
        "Fever",
        "Stomachache",
        "Wound dressing",
        "Allergy",
        "Toothache",
        "Anxiety attack",
    ]
    
    for complaint in test_cases:
        category = categorize_complaint(complaint)
        print(f"{complaint:30s} → {category}")
    
    print("=" * 70)
