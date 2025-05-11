import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

class AggressiveBertClassifier(nn.Module):
    def __init__(self, output_dim=20, dropout_prob=0.3):
        super(AggressiveBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased',
            attn_implementation="eager"
        )
        
        # Simplified classifier to match the saved model
        self.classifier = nn.Linear(768, output_dim)  # Single linear layer
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.classifier.bias, 0.1)  # Slight positive bias

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True
        )
        
        # Use the [CLS] token embedding
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        
        # Classification
        logits = self.classifier(pooled)
        # Return a dummy attention_weights to maintain compatibility
        attention_weights = torch.zeros_like(logits)
        return logits, attention_weights

class ThreatAnalyzer:
    def __init__(self, model_path: str = "/content/fine_tuned_bert_multilabel.pt"):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = AggressiveBertClassifier()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("Warning: Model path not found or invalid - using random initialization")
        
        self.model.eval()
        
        # Adjusted category definitions to match output_dim=20
        self.categories = {
            'violence': {
                'classes': 7,  # Extended to 7 classes to account for extra outputs
                'thresholds': [0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02],  # Adjusted thresholds
                'weights': [0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],  # Adjusted weights
                'descriptions': [
                    "No violence",
                    "Mild violent language",
                    "Moderate violent intent",
                    "Serious violent threat",
                    "Severe violent threat",
                    "Critical violent threat",
                    "Catastrophic violent threat"
                ]
            },
            'genocide': {
                'classes': 5,
                'thresholds': [0.12, 0.1, 0.08, 0.06],
                'weights': [0, 1.2, 1.8, 2.5, 4.0],
                'descriptions': [
                    "No genocide rhetoric",
                    "Mild exclusionary language",
                    "Moderate dehumanization",
                    "Serious elimination rhetoric",
                    "Explicit genocide advocacy"
                ]
            },
            'hatespeech': {
                'classes': 3,
                'thresholds': [0.15, 0.1],
                'weights': [0, 1.5, 3.0],
                'descriptions': [
                    "No hate speech",
                    "Problematic language",
                    "Explicit hate speech"
                ]
            },
            # Placeholder for the remaining 5 classes (adjust based on your data)
            'other': {
                'classes': 5,
                'thresholds': [0.15, 0.12, 0.1, 0.08],
                'weights': [0, 1.0, 2.0, 3.0, 4.0],
                'descriptions': [
                    "No other concern",
                    "Mild concern",
                    "Moderate concern",
                    "Serious concern",
                    "Critical concern"
                ]
            }
        }

    def load_model(self, model_path: str):
        try:
            state = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        with torch.no_grad():
            logits, attention = self.model(**inputs)
            probs = F.softmax(logits, dim=-1).squeeze()
        
        return self._interpret_probs(probs.numpy(), attention.numpy(), inputs)

    def _interpret_probs(self, probs: np.ndarray, attention: np.ndarray, inputs: Dict) -> Dict:
        results = {}
        start_idx = 0
        
        # Process each category
        for cat, info in self.categories.items():
            end_idx = start_idx + info['classes']
            cat_probs = probs[start_idx:end_idx]
            
            # More sensitive classification
            pred_class, confidence = self._classify_category(cat_probs, info)
            
            results[cat] = {
                'class': pred_class,
                'confidence': confidence,
                'description': info['descriptions'][pred_class],
                'probabilities': [float(f"{p:.4f}") for p in cat_probs]
            }
            start_idx = end_idx
        
        # Calculate aggressive risk score
        risk = self._calculate_risk(results)
        
        # Analyze attention (simplified since there's no attention mechanism)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        top_tokens = self._get_top_attention(tokens, attention)
        
        return {
            'predictions': results,
            'risk_assessment': risk,
            'attention': top_tokens
        }

    def _classify_category(self, probs: np.ndarray, info: Dict) -> Tuple[int, float]:
        """More aggressive classification with lower thresholds"""
        pred_class = np.argmax(probs)
        max_prob = probs[pred_class]
        
        # If predicted class is 0 (neutral), we still check if any other class is above threshold
        if pred_class == 0:
            max_other = np.max(probs[1:])
            threshold = info['thresholds'][0] if info['thresholds'] else 0.1
            
            if max_other > threshold:
                # Override prediction if any class is above threshold
                pred_class = np.argmax(probs[1:]) + 1
                max_prob = probs[pred_class]
        
        # Calculate confidence more aggressively
        if pred_class == 0:
            confidence = 1 - np.max(probs[1:])
        else:
            min_threshold = info['thresholds'][min(pred_class-1, len(info['thresholds'])-1)]
            confidence = min(1.0, max_prob / min_threshold)
        
        return pred_class, float(confidence)

    def _calculate_risk(self, predictions: Dict) -> Dict:
        """More sensitive risk calculation"""
        total = 0
        max_possible = 0
        
        for cat, data in predictions.items():
            info = self.categories[cat]
            weight = info['weights'][data['class']]
            total += weight * data['confidence']
            max_possible += info['weights'][-1]  # Max weight for this category
        
        score = total / max_possible if max_possible > 0 else 0
        
        # More sensitive risk levels
        if score > 0.8:
            level = "CRITICAL"
        elif score > 0.6:
            level = "HIGH"
        elif score > 0.4:
            level = "MODERATE"
        elif score > 0.2:
            level = "LOW"
        else:
            level = "MINIMAL"
        
        return {'score': float(score), 'level': level}

    def _get_top_attention(self, tokens: List[str], attention: np.ndarray) -> List[Tuple[str, float]]:
        """Simplified attention analysis since there's no attention mechanism"""
        return [(token, 0.0) for token in tokens[:5]]

    def analyze(self, text: str) -> str:
        """Generate comprehensive analysis report"""
        result = self.predict(text)
        
        report = [
            f"\n{'='*80}",
            f"THREAT ANALYSIS REPORT: '{text}'",
            f"\nOVERALL RISK: {result['risk_assessment']['level']} (Score: {result['risk_assessment']['score']:.2f})",
            "\nDETAILED FINDINGS:"
        ]
        
        for cat, data in result['predictions'].items():
            report.extend([
                f"\n--- {cat.upper()} ---",
                f"Classification: {data['class']} - {data['description']}",
                f"Confidence: {data['confidence']:.2f}",
                "Probabilities:"
            ] + [f"  {i}: {p:.4f}" for i, p in enumerate(data['probabilities'])])
        
        report.extend([
            "\nATTENTION ANALYSIS:",
            "Most Important Words (Simplified):"
        ] + [f"  '{word}': {weight:.4f}" for word, weight in result['attention']])
        
        report.append(f"\n{'='*80}")
        return "\n".join(report)

import os

# Initialize the threat analyzer with the model weights
threat_analyzer = ThreatAnalyzer(model_path="/content/fine_tuned_bert_multilabel.pt" if os.path.exists("/content/fine_tuned_bert_multilabel.pt") else None)

def generate_violence_report(tracking_summary: str) -> str:
    """Generate a violence classification report using the ThreatAnalyzer based on tracking summary."""
    if not tracking_summary or tracking_summary.strip() == "":
        return "No violence report generated due to lack of tracking summary."
    
    # Use the threat analyzer to generate the report
    report = threat_analyzer.analyze(tracking_summary)
    print("Violence report generated successfully.")
    return report

# Export the generator function for use in routes.py
violence_report_generator = generate_violence_report