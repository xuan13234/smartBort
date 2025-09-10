# chatbot_evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu
import nltk
from collections import Counter
import json
import ast

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def evaluate_chatbot_performance(log_file="data/chatbot_logs.csv", intents_file="data/intents.json"):
    """
    Comprehensive evaluation of chatbot performance
    """
    # Load data
    df = pd.read_csv(log_file)
    
    # Load intents for ground truth (if available)
    with open(intents_file, "r") as f:
        intents_data = json.load(f)
    
    # 1. Intent Classification Evaluation
    print("="*50)
    print("INTENT CLASSIFICATION EVALUATION")
    print("="*50)
    
    if 'predicted_tag' in df.columns and 'true_tag' in df.columns:
        # If we have ground truth labels
        y_true = df['true_tag'].dropna()
        y_pred = df['predicted_tag'].dropna()
        
        # Calculate precision, recall, f1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        # Confusion matrix
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - Intent Classification')
        plt.ylabel('True Intent')
        plt.xlabel('Predicted Intent')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('intent_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
    else:
        print("No ground truth labels available for intent classification evaluation")
    
    # 2. Response Quality Evaluation (BLEU Score)
    print("\n" + "="*50)
    print("RESPONSE QUALITY EVALUATION (BLEU)")
    print("="*50)
    
    if 'user_input' in df.columns and 'response' in df.columns:
        # Calculate BLEU scores
        smoothie = SmoothingFunction().method4
        bleu_scores = []
        
        for _, row in df.iterrows():
            reference = [str(row["user_input"]).split()]
            candidate = str(row["response"]).split()
            try:
                score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
                bleu_scores.append(score)
            except:
                bleu_scores.append(0)
        
        df['bleu_score'] = bleu_scores
        avg_bleu = np.mean(bleu_scores)
        
        print(f"Average BLEU Score: {avg_bleu:.3f}")
        print(f"BLEU Score Distribution: Min={min(bleu_scores):.3f}, "
              f"Max={max(bleu_scores):.3f}, Std={np.std(bleu_scores):.3f}")
        
        # Plot BLEU score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(bleu_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(avg_bleu, color='red', linestyle='dashed', linewidth=1, 
                   label=f'Mean: {avg_bleu:.3f}')
        plt.xlabel('BLEU Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of BLEU Scores')
        plt.legend()
        plt.savefig('bleu_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. User Satisfaction Evaluation
    print("\n" + "="*50)
    print("USER SATISFACTION EVALUATION")
    print("="*50)
    
    if 'feedback' in df.columns:
        feedback_counts = df['feedback'].value_counts()
        total_feedback = feedback_counts.sum()
        
        if total_feedback > 0:
            positive_feedback = feedback_counts.get('yes', 0) + feedback_counts.get(1, 0)
            negative_feedback = feedback_counts.get('no', 0) + feedback_counts.get(0, 0)
            
            satisfaction_rate = positive_feedback / total_feedback if total_feedback > 0 else 0
            
            print(f"Total Feedback: {total_feedback}")
            print(f"Positive Feedback: {positive_feedback}")
            print(f"Negative Feedback: {negative_feedback}")
            print(f"Satisfaction Rate: {satisfaction_rate:.3f}")
            
            # Plot feedback distribution
            labels = ['Positive', 'Negative']
            sizes = [positive_feedback, negative_feedback]
            colors = ['#66b3ff', '#ff9999']
            
            plt.figure(figsize=(8, 8))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('User Feedback Distribution')
            plt.savefig('user_feedback_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("No user feedback data available")
    else:
        print("No feedback column in the log file")
    
    # 4. Additional Metrics
    print("\n" + "="*50)
    print("ADDITIONAL PERFORMANCE METRICS")
    print("="*50)
    
    # Response time analysis (if timestamp available)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['response_time'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
        avg_response_time = df['response_time'].mean()
        print(f"Average Response Time: {avg_response_time:.2f} seconds")
    
    # Confidence analysis (if confidence available)
    if 'confidence' in df.columns:
        avg_confidence = df['confidence'].mean()
        print(f"Average Confidence Score: {avg_confidence:.3f}")
    
    # Most common intents
    if 'predicted_tag' in df.columns:
        top_intents = df['predicted_tag'].value_counts().head(5)
        print("\nTop 5 Most Common Intents:")
        for intent, count in top_intents.items():
            print(f"  {intent}: {count} interactions")
    
    # Save comprehensive report
    report_data = {
        'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_interactions': len(df),
        'avg_bleu_score': avg_bleu if 'bleu_scores' in locals() else None,
        'satisfaction_rate': satisfaction_rate if 'satisfaction_rate' in locals() else None,
        'avg_response_time': avg_response_time if 'avg_response_time' in locals() else None,
        'avg_confidence': avg_confidence if 'avg_confidence' in locals() else None
    }
    
    report_df = pd.DataFrame([report_data])
    report_df.to_csv('chatbot_performance_report.csv', index=False)
    
    print("\nEvaluation complete! Results saved to:")
    print("- intent_confusion_matrix.png")
    print("- bleu_score_distribution.png")
    print("- user_feedback_distribution.png")
    print("- chatbot_performance_report.csv")

if __name__ == "__main__":
    evaluate_chatbot_performance()