import sys
from text_pipeline import TextPipeline
from fusion_engine import FusionEngine

def run_simulation():
    # Initialize components
    text_pipe = TextPipeline()
    fusion = FusionEngine()
    
    print("Loading RoBERTa model...")
    text_pipe.load_model()
    
    # Scenarios: (Name, Emotion, Text, Simulated Visual Score, Simulated Visual Conf)
    scenarios = [
        ("Congruent Positive", "happy", "I am thrilled about this!", 1.0, 0.95),
        ("Congruent Negative", "sad", "This is a terrible experience.", -0.7, 0.90),
        ("Sarcasm/Masking", "angry", "Oh, what a wonderful day.", -1.0, 0.95),
        ("Text-Dominant", "neutral", "The product quality is excellent.", 0.0, 0.20) # Low confidence on neutral face
    ]
    
    # Header
    print("\n" + "="*95)
    print(f"{'Scenario':<20} | {'Face (Prob)':<12} | {'Textual Input':<36} | {'Conflict?':<10} | {'Fused Sentiment Result'}")
    print("="*95)
    
    for name, emotion, text, vis_score, vis_conf in scenarios:
        # Run text analysis
        text_pipe.analyze(text)
        txt_score, txt_conf = text_pipe.get_sentiment_score()
        
        # Run dynamic fusion
        final_score, final_conf, wv, wt = fusion.dynamic_fusion(vis_score, vis_conf, txt_score, txt_conf)
        conflict, conflict_msg = fusion.detect_conflict(vis_score, txt_score)
        label = fusion.classify_sentiment(final_score)
        
        conflict_str = "Yes" if conflict else "No"
        fused_result = f"{label} ({final_score:+.2f}, Wv:{wv*100:.0f}%, Wt:{wt*100:.0f}%)"
        
        print(f"{name:<20} | {emotion:<12} | {text:<36} | {conflict_str:<10} | {fused_result}")
        
    print("="*95 + "\n")

if __name__ == "__main__":
    run_simulation()
