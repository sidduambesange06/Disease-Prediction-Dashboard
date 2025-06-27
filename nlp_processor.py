from transformers import pipeline
import re

class SymptomExtractor:
    def __init__(self, symptom_list):
        """
        Initialize the symptom extractor with a list of valid symptoms.
        
        Args:
            symptom_list (list): List of valid symptoms from the model
        """
        self.symptom_list = symptom_list
        # Load NER pipeline from HuggingFace
        self.ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all")
        
        # Create a mapping of common symptom phrases to standardized symptoms
        self.symptom_mapping = self._create_symptom_mapping()
        
    def _create_symptom_mapping(self):
        """Create a mapping of various ways symptoms might be described to standardized symptoms."""
        mapping = {}
        
        # Convert symptom names to more natural language variations
        for symptom in self.symptom_list:
            # Convert underscores to spaces
            natural_symptom = symptom.replace('_', ' ')
            mapping[natural_symptom] = symptom
            
            # Add common variations
            if "pain" in natural_symptom:
                mapping[natural_symptom.replace("pain", "ache")] = symptom
            
            # Add plural forms
            if not natural_symptom.endswith('s'):
                mapping[natural_symptom + 's'] = symptom
                
        # Add some common symptom descriptions that might not match directly
        common_mappings = {
            "can't sleep": "insomnia",
            "trouble sleeping": "insomnia",
            "runny nose": "nasal_discharge",
            "stuffy nose": "nasal_congestion",
            "throwing up": "vomiting",
            "feel sick": "nausea",
            "feel nauseous": "nausea",
            "stomach ache": "abdominal_pain",
            "belly pain": "abdominal_pain",
            "sore throat": "throat_pain",
            "high temperature": "fever",
            "elevated temperature": "fever",
            "feeling tired": "fatigue",
            "exhausted": "fatigue",
            "no energy": "fatigue",
            "dizzy": "dizziness",
            "feeling dizzy": "dizziness",
            "short of breath": "shortness_of_breath",
            "hard to breathe": "shortness_of_breath",
            "trouble breathing": "shortness_of_breath",
            "chest pain": "chest_pain",
            "heart pain": "chest_pain",
        }
        
        for common, standard in common_mappings.items():
            if standard in self.symptom_list:
                mapping[common] = standard
            
        return mapping
    
    def _normalize_text(self, text):
        """Normalize text by converting to lowercase and removing punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text
    
    def _extract_symptoms_from_ner(self, text):
        """Extract medical entities from text using the NER pipeline."""
        entities = self.ner_pipeline(text)
        
        # Filter for relevant entity types (symptoms, diseases, etc.)
        relevant_entities = []
        for entity in entities:
            # The model tags various medical entities
            if entity['entity'].startswith('B-') or entity['entity'].startswith('I-'):
                relevant_entities.append(entity['word'].lower())
        
        return relevant_entities
    
    def _match_to_known_symptoms(self, extracted_terms):
        """Match extracted terms to known symptoms in our model."""
        matched_symptoms = set()
        
        for term in extracted_terms:
            # Check if the term directly matches a symptom
            if term in self.symptom_mapping:
                matched_symptoms.add(self.symptom_mapping[term])
                continue
                
            # Check if the term is part of any symptom phrase
            for symptom_phrase, standard_symptom in self.symptom_mapping.items():
                if term in symptom_phrase.split() or symptom_phrase in term:
                    matched_symptoms.add(standard_symptom)
        
        # Convert to list for consistency
        return list(matched_symptoms)
    
    def _rule_based_extraction(self, text):
        """Use rule-based approach to extract symptoms."""
        normalized_text = self._normalize_text(text)
        matched_symptoms = set()
        
        # Check for each symptom phrase in the text
        for symptom_phrase, standard_symptom in self.symptom_mapping.items():
            if symptom_phrase in normalized_text:
                matched_symptoms.add(standard_symptom)
        
        return list(matched_symptoms)
    
    def extract_symptoms(self, text):
        """
        Extract symptoms from free text using both NER and rule-based approaches.
        
        Args:
            text (str): Free text describing symptoms
            
        Returns:
            list: List of standardized symptoms found in the text
        """
        # Normalize the text
        normalized_text = self._normalize_text(text)
        
        # Extract entities using NER
        ner_entities = self._extract_symptoms_from_ner(text)
        
        # Match NER entities to known symptoms
        ner_symptoms = self._match_to_known_symptoms(ner_entities)
        
        # Use rule-based approach as backup
        rule_symptoms = self._rule_based_extraction(normalized_text)
        
        # Combine results from both approaches
        all_symptoms = list(set(ner_symptoms + rule_symptoms))
        
        # Filter to ensure all returned symptoms are in our symptom list
        valid_symptoms = [s for s in all_symptoms if s in self.symptom_list]
        
        return valid_symptoms

# Function to test the symptom extractor
def test_extractor(extractor, test_texts):
    """Test the symptom extractor with sample texts."""
    for text in test_texts:
        print(f"Text: {text}")
        symptoms = extractor.extract_symptoms(text)
        print(f"Extracted symptoms: {symptoms}")
        print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Sample symptom list (replace with actual symptoms from your model)
    sample_symptoms = ["fever", "headache", "cough", "fatigue", "sore_throat", 
                    "shortness_of_breath", "chest_pain", "abdominal_pain"]
    
    extractor = SymptomExtractor(sample_symptoms)
    
    test_texts = [
        "I've been having a bad headache and fever for the past two days.",
        "My throat is sore and I'm coughing a lot.",
        "I feel tired all the time and sometimes it's hard to breathe.",
        "I have pain in my chest and stomach."
    ]
    
    test_extractor(extractor, test_texts)