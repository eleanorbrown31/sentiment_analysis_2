import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="ML Sentiment Analysis: Presentation Feedback", layout="wide")

# Load training data
@st.cache_data
def load_training_data():
    """Load training data from CSV file"""
    try:
        # Try to load from uploaded CSV
        df = pd.read_csv('training_data.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ training_data.csv file not found. Please upload the training data file.")
        return None

# Data splitting and validation explanation
def explain_data_splitting():
    """Educational component explaining why we split data"""
    st.markdown("""
    ### 🎯 Why Do We Split Our Data?
    
    **The Problem**: If we test our model on the same data we trained it on, it's like giving students the exact same exam they studied from - they'll score perfectly but learn nothing!
    
    **The Solution**: Three-way data split
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🏋️ Training Set (60%)**
        - Model learns patterns here
        - "Study materials"
        - Larger = better learning
        """)
    
    with col2:
        st.markdown("""
        **🔍 Validation Set (20%)**
        - Tests during development
        - "Practice exam"
        - Helps tune the model
        """)
    
    with col3:
        st.markdown("""
        **🧪 Test Set (20%)**
        - Final performance check
        - "Real exam"
        - Never seen before!
        """)

# Create comprehensive data splits
@st.cache_data
def create_data_splits(df):
    """Create train/validation/test splits with educational tracking"""
    if df is None:
        return None
        
    texts = df['text'].tolist()
    labels = df['sentiment'].tolist()
    
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Second split: 75% train, 25% validation (of the remaining 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'train_df': pd.DataFrame({'text': X_train, 'sentiment': y_train}),
        'val_df': pd.DataFrame({'text': X_val, 'sentiment': y_val}),
        'test_df': pd.DataFrame({'text': X_test, 'sentiment': y_test})
    }

# Enhanced evaluation functions
def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    
    return metrics

def plot_validation_curve_explanation(X_train, y_train, X_val, y_val):
    """Show why validation set prevents overfitting using vocabulary size"""
    st.subheader("🎓 Understanding Overfitting with Validation Curves")
    
    st.markdown("""
    **Watch what happens as we give our model more vocabulary words to learn from:**
    - **Few words**: Model can't learn enough (underfitting)
    - **Too many words**: Model memorizes noise and rare words (overfitting)
    - **Just right**: Model learns useful patterns that generalize
    
    **The validation curve shows us the "sweet spot"!**
    """)
    
    # Test different vocabulary sizes (this is much more intuitive!)
    vocab_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    
    train_scores = []
    val_scores = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, vocab_size in enumerate(vocab_sizes):
        status_text.text(f"Testing vocabulary size: {vocab_size} words...")
        
        # Create vectorizer with specific vocabulary size
        vectorizer = TfidfVectorizer(
            max_features=vocab_size,
            stop_words='english',
            ngram_range=(1,1)  # Keep it simple - just single words
        )
        
        # Transform data
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_vec, y_train)
        
        # Calculate scores
        train_score = model.score(X_train_vec, y_train)
        val_score = model.score(X_val_vec, y_val)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        
        progress_bar.progress((i + 1) / len(vocab_sizes))
    
    status_text.text("Validation curve complete!")
    
    # Plot validation curve
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=vocab_sizes,
        y=train_scores,
        mode='lines+markers',
        name='Training Accuracy',
        line=dict(color='blue', width=4),
        marker=dict(size=10, symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=vocab_sizes,
        y=val_scores,
        mode='lines+markers',
        name='Validation Accuracy',
        line=dict(color='red', width=4),
        marker=dict(size=10, symbol='square')
    ))
    
    # Find optimal point
    best_idx = np.argmax(val_scores)
    optimal_vocab = vocab_sizes[best_idx]
    
    fig.add_vline(x=optimal_vocab, line_dash="dash", line_color="green", line_width=3,
                  annotation_text=f"Sweet Spot: {optimal_vocab} words")
    
    # Add regions
    fig.add_vrect(x0=0, x1=200, fillcolor="yellow", opacity=0.2, annotation_text="Underfitting")
    fig.add_vrect(x0=2000, x1=5000, fillcolor="orange", opacity=0.2, annotation_text="Overfitting")
    
    fig.update_layout(
        title='Validation Curve: Finding the Right Vocabulary Size',
        xaxis_title='Vocabulary Size (number of words model can learn)',
        yaxis_title='Accuracy',
        height=500,
        font=dict(size=14)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Educational explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### 📊 What We Discovered:
        
        **Best vocabulary size**: {optimal_vocab} words  
        **Training accuracy**: {train_scores[best_idx]:.3f}  
        **Validation accuracy**: {val_scores[best_idx]:.3f}
        
        **🎯 Key Insight**: Training accuracy keeps going up, but validation accuracy peaks and then drops!
        """)
    
    with col2:
        st.markdown("""
        ### 🤔 Why This Happens:
        
        **Too few words (underfitting)**:
        - Model can't express complex patterns
        - Both training and validation scores are low
        
        **Too many words (overfitting)**:
        - Model memorizes rare, meaningless words
        - Training score high, validation score drops
        
        **Just right**:
        - Model learns useful, generalizable patterns
        """)
    
    with st.expander("🎓 The Big Lesson"):
        st.markdown(f"""
        **This is exactly why we need validation data!**
        
        - If we only looked at **training accuracy**, we'd think {vocab_sizes[-1]} words is best ({train_scores[-1]:.3f} accuracy)
        - But the **validation accuracy** tells the truth - it's only {val_scores[-1]:.3f}!
        - The validation set acts as an "honest judge" that catches overfitting
        
        **Without validation data**, we'd build overconfident models that fail in the real world.
        
        **Real-world lesson**: Always test your model on data it hasn't seen during training!
        """)

def analyze_misclassified_examples(y_true, y_pred, texts, probabilities):
    """Detailed analysis of what the model gets wrong"""
    st.subheader("🔍 Detailed Analysis: What Did We Get Wrong?")
    
    # Find misclassified examples
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) == 0:
        st.success("🎉 Perfect! No misclassified examples!")
        return
    
    # Create detailed analysis dataframe
    misclassified_df = pd.DataFrame({
        'text': [texts[i] for i in misclassified_indices],
        'true_sentiment': ['Positive' if y_true[i] == 1 else 'Negative' for i in misclassified_indices],
        'predicted_sentiment': ['Positive' if y_pred[i] == 1 else 'Negative' for i in misclassified_indices],
        'confidence': [max(probabilities[i]) * 100 for i in misclassified_indices],
        'positive_probability': [probabilities[i][1] * 100 for i in misclassified_indices]
    })
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Misclassified", len(misclassified_indices))
    
    with col2:
        false_positives = sum((y_true[misclassified_indices] == 0) & (y_pred[misclassified_indices] == 1))
        st.metric("False Positives", false_positives)
    
    with col3:
        false_negatives = sum((y_true[misclassified_indices] == 1) & (y_pred[misclassified_indices] == 0))
        st.metric("False Negatives", false_negatives)
    
    # Interactive analysis
    tab1, tab2, tab3 = st.tabs(["🔍 Browse Mistakes", "📊 Pattern Analysis", "🎯 Confidence Analysis"])
    
    with tab1:
        st.markdown("**Click through the misclassified examples to understand the model's mistakes:**")
        
        # Example selector
        if len(misclassified_df) > 0:
            example_idx = st.selectbox(
                "Select a misclassified example:",
                range(len(misclassified_df)),
                format_func=lambda x: f"Example {x+1}: {misclassified_df.iloc[x]['text'][:50]}..."
            )
            
            selected_example = misclassified_df.iloc[example_idx]
            
            # Display selected example with highlighting
            true_color = "green" if selected_example['true_sentiment'] == 'Positive' else "red"
            pred_color = "green" if selected_example['predicted_sentiment'] == 'Positive' else "red"
            
            st.markdown(f"""
            <div style="border: 2px solid #ccc; border-radius: 10px; padding: 20px; margin: 10px 0;">
                <h4>📝 "{selected_example['text']}"</h4>
                <p><strong>Actually:</strong> <span style="color:{true_color}; font-weight:bold;">{selected_example['true_sentiment']}</span></p>
                <p><strong>Predicted:</strong> <span style="color:{pred_color}; font-weight:bold;">{selected_example['predicted_sentiment']}</span></p>
                <p><strong>Confidence:</strong> {selected_example['confidence']:.1f}%</p>
                <p><strong>Positive Probability:</strong> {selected_example['positive_probability']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Analysis questions
            st.markdown("**🤔 Why might the model have made this mistake?**")
            with st.expander("Click for analysis hints"):
                text = selected_example['text'].lower()
                
                hints = []
                if 'but' in text or 'however' in text:
                    hints.append("• Contains contradictory language ('but', 'however')")
                if any(word in text for word in ['okay', 'fine', 'alright']):
                    hints.append("• Uses neutral words that could be ambiguous")
                if selected_example['confidence'] < 60:
                    hints.append(f"• Low confidence ({selected_example['confidence']:.1f}%) suggests the model was uncertain")
                if 'not' in text or "n't" in text:
                    hints.append("• Contains negation which can be tricky for models to handle")
                if len(text.split()) < 10:
                    hints.append("• Short text with limited context")
                
                if hints:
                    for hint in hints:
                        st.markdown(hint)
                else:
                    st.markdown("• This appears to be a challenging example even for humans!")
    
    with tab2:
        st.markdown("**Common patterns in misclassified examples:**")
        
        # Analyze common words in misclassified examples
        misclassified_texts = misclassified_df['text'].str.lower()
        all_words = []
        for text in misclassified_texts:
            words = re.findall(r'\b\w+\b', text)
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(10)
        
        if common_words:
            words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
            
            fig = px.bar(words_df, x='Word', y='Frequency', 
                        title='Most Common Words in Misclassified Examples')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Pattern analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**False Positives (Predicted Positive, Actually Negative):**")
            fp_examples = misclassified_df[
                (misclassified_df['true_sentiment'] == 'Negative') & 
                (misclassified_df['predicted_sentiment'] == 'Positive')
            ]
            if len(fp_examples) > 0:
                for i, (_, row) in enumerate(fp_examples.head(3).iterrows()):
                    st.markdown(f"• {row['text'][:80]}...")
            else:
                st.markdown("No false positives in this sample!")
        
        with col2:
            st.markdown("**False Negatives (Predicted Negative, Actually Positive):**")
            fn_examples = misclassified_df[
                (misclassified_df['true_sentiment'] == 'Positive') & 
                (misclassified_df['predicted_sentiment'] == 'Negative')
            ]
            if len(fn_examples) > 0:
                for i, (_, row) in enumerate(fn_examples.head(3).iterrows()):
                    st.markdown(f"• {row['text'][:80]}...")
            else:
                st.markdown("No false negatives in this sample!")
    
    with tab3:
        st.markdown("**How confident was the model in its wrong predictions?**")
        
        # Confidence distribution of misclassified examples
        fig = px.histogram(misclassified_df, x='confidence', nbins=10,
                          title='Confidence Distribution of Misclassified Examples',
                          labels={'confidence': 'Confidence (%)', 'count': 'Number of Examples'})
        fig.add_vline(x=50, line_dash="dash", line_color="red", 
                     annotation_text="50% (Random Guess)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights about confidence
        low_confidence = misclassified_df[misclassified_df['confidence'] < 60]
        high_confidence = misclassified_df[misclassified_df['confidence'] >= 80]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Low Confidence Mistakes (<60%)", len(low_confidence))
            st.caption("These are 'uncertain' mistakes - the model wasn't sure")
        
        with col2:
            st.metric("High Confidence Mistakes (≥80%)", len(high_confidence))
            st.caption("These are 'confident' mistakes - more concerning!")
        
        if len(high_confidence) > 0:
            st.markdown("**Examples where the model was confidently wrong:**")
            for _, row in high_confidence.head(2).iterrows():
                st.markdown(f"• \"{row['text'][:60]}...\" ({row['confidence']:.1f}% confident)")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'data_splits' not in st.session_state:
    st.session_state.data_splits = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'training_params' not in st.session_state:
    st.session_state.training_params = None

# App title and description
st.title("🧠 ML Sentiment Analysis: Virtual Presentation Feedback")

st.markdown("""
**Learn machine learning by analyzing feedback on virtual ML presentations!**

This demo teaches core ML concepts through a real-world task: automatically classifying presentation feedback as positive or negative.

**Key Learning Goals:**
- Why we split data into train/validation/test sets
- How validation prevents overfitting
- What happens when models make mistakes
- Professional ML evaluation techniques
""")

# Load data
training_df = load_training_data()

if training_df is None:
    st.stop()

# Data overview
with st.expander("📊 View Training Data Sample"):
    st.markdown(f"**Dataset**: {len(training_df)} feedback comments on virtual ML presentations")
    
    col1, col2 = st.columns(2)
    with col1:
        positive_count = len(training_df[training_df['sentiment'] == 1])
        st.metric("Positive Feedback", positive_count)
    with col2:
        negative_count = len(training_df[training_df['sentiment'] == 0])
        st.metric("Negative Feedback", negative_count)
    
    # Show sample of each type
    sample_positive = training_df[training_df['sentiment'] == 1].sample(5)
    sample_negative = training_df[training_df['sentiment'] == 0].sample(5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sample Positive Feedback:**")
        for text in sample_positive['text']:
            st.markdown(f"✅ {text}")
    
    with col2:
        st.markdown("**Sample Negative Feedback:**")
        for text in sample_negative['text']:
            st.markdown(f"❌ {text}")

st.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Data Splitting", 
    "🏋️ Model Training", 
    "🧪 Testing & Analysis", 
    "🔮 Try It Yourself"
])

# Tab 1: Data Splitting Education
with tab1:
    st.header("📚 Understanding Data Splits")
    
    explain_data_splitting()
    
    if st.button("🎯 Create Data Splits", type="primary"):
        with st.spinner("Splitting data..."):
            splits = create_data_splits(training_df)
            st.session_state.data_splits = splits
        
        st.success("✅ Data split successfully!")
    
    if st.session_state.data_splits:
        splits = st.session_state.data_splits
        
        st.markdown("### 📊 Your Data Splits")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Set", f"{len(splits['X_train'])} examples")
            with st.expander("View Training Sample"):
                st.dataframe(splits['train_df'].head(10))
        
        with col2:
            st.metric("Validation Set", f"{len(splits['X_val'])} examples")
            with st.expander("View Validation Sample"):
                st.dataframe(splits['val_df'].head(10))
        
        with col3:
            st.metric("Test Set", f"{len(splits['X_test'])} examples")
            with st.expander("View Test Sample"):
                st.markdown("🔒 **Locked until testing phase!**")
                st.markdown("We don't look at test data until final evaluation")

# Tab 2: Model Training
with tab2:
    st.header("🏋️ Model Training & Validation")
    
    if st.session_state.data_splits is None:
        st.warning("⚠️ Please create data splits first in the 'Data Splitting' tab!")
    else:
        splits = st.session_state.data_splits
        
        st.markdown("### 🔧 Training Configuration")
        
        st.markdown("**Experiment with these simple settings to see how they affect learning:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            vocab_size = st.slider("📚 Vocabulary Size", 
                                 min_value=100, max_value=2000, value=1000, step=100,
                                 help="How many different words can the model learn? More isn't always better!")
            
            min_word_freq = st.slider("🔢 Minimum Word Frequency", 
                                    min_value=1, max_value=5, value=2,
                                    help="Ignore words that appear fewer than this many times")
        
        with col2:
            training_size = st.slider("🎯 Training Set Size", 
                                    min_value=0.3, max_value=1.0, value=1.0, step=0.1,
                                    help="Use this fraction of available training data")
            
            remove_stop_words = st.checkbox("🚫 Remove Stop Words", value=True,
                                          help="Remove common words like 'the', 'and', 'is'")
        
        # Show what these settings mean
        with st.expander("💡 What do these settings do?"):
            st.markdown(f"""
            **📚 Vocabulary Size ({vocab_size} words):**
            - Too small: Model can't learn enough patterns (underfitting)
            - Too large: Model memorizes rare words (overfitting)
            
            **🔢 Minimum Word Frequency ({min_word_freq}):**
            - Higher numbers ignore rare words that might be noise
            - Lower numbers include more words but risk overfitting
            
            **🎯 Training Set Size ({training_size:.0%}):**
            - More data usually means better learning
            - But shows diminishing returns after a point
            
            **🚫 Stop Words ({'Removed' if remove_stop_words else 'Included'}):**
            - Common words like 'the', 'and' usually don't help with sentiment
            - But sometimes they matter in context!
            """)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                
                # Use specified fraction of training data
                if training_size < 1.0:
                    train_sample_size = int(len(splits['X_train']) * training_size)
                    X_train_sample = splits['X_train'][:train_sample_size]
                    y_train_sample = splits['y_train'][:train_sample_size]
                else:
                    X_train_sample = splits['X_train']
                    y_train_sample = splits['y_train']
                
                # Create vectorizer with chosen settings
                vectorizer = TfidfVectorizer(
                    max_features=vocab_size,
                    stop_words='english' if remove_stop_words else None,
                    min_df=min_word_freq,
                    ngram_range=(1,1)  # Keep it simple - just single words
                )
                
                # Transform training data
                X_train_vec = vectorizer.fit_transform(X_train_sample)
                
                # Train model (keep it simple - no regularization parameter)
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train_vec, y_train_sample)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                st.session_state.training_complete = True
                st.session_state.training_params = {
                    'vocab_size': vocab_size,
                    'min_word_freq': min_word_freq,
                    'training_size': training_size,
                    'remove_stop_words': remove_stop_words,
                    'actual_training_examples': len(X_train_sample)
                }
            
            st.success(f"✅ Model trained successfully using {len(X_train_sample)} examples!")
        
        # Show validation curve if model is trained
        if st.session_state.training_complete:
            plot_validation_curve_explanation(
                splits['X_train'], splits['y_train'],
                splits['X_val'], splits['y_val']
            )
            
            # Show feature importance
            st.subheader("🔍 What Did the Model Learn?")
            
            # Show training summary
            params = st.session_state.training_params
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                ### 📋 Training Summary:
                - **Training examples used**: {params['actual_training_examples']}
                - **Vocabulary size**: {params['vocab_size']} words
                - **Min word frequency**: {params['min_word_freq']}
                - **Stop words**: {'Removed' if params['remove_stop_words'] else 'Included'}
                """)
            
            with col2:
                # Quick validation check
                X_val_vec = st.session_state.vectorizer.transform(splits['X_val'])
                val_accuracy = st.session_state.model.score(X_val_vec, splits['y_val'])
                st.metric("Validation Accuracy", f"{val_accuracy:.3f}")
                st.caption("How well the model performs on unseen validation data")
            
            # Feature importance analysis
            feature_names = st.session_state.vectorizer.get_feature_names_out()
            coefficients = st.session_state.model.coef_[0]
            
            st.markdown("### 📊 Most Important Words the Model Learned:")
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients
            }).sort_values('coefficient', key=abs, ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🟢 Most Positive Words:**")
                positive_features = feature_importance[feature_importance['coefficient'] > 0].head(8)
                for _, row in positive_features.iterrows():
                    st.markdown(f"• **{row['feature']}**: +{row['coefficient']:.3f}")
                st.caption("Words that make the model predict 'positive feedback'")
            
            with col2:
                st.markdown("**🔴 Most Negative Words:**")
                negative_features = feature_importance[feature_importance['coefficient'] < 0].head(8)
                for _, row in negative_features.iterrows():
                    st.markdown(f"• **{row['feature']}**: {row['coefficient']:.3f}")
                st.caption("Words that make the model predict 'negative feedback'")
            
            # Show vocabulary size impact
            actual_vocab = len(feature_names)
            with st.expander(f"🔍 Vocabulary Analysis ({actual_vocab} words actually used)"):
                st.markdown(f"""
                **You set max vocabulary to {params['vocab_size']} words, but the model actually uses {actual_vocab} words.**
                
                This happens because:
                - Words appearing less than {params['min_word_freq']} times were filtered out
                - {'Stop words were removed' if params['remove_stop_words'] else 'Stop words were included'}
                - The dataset might not have enough unique words to reach the maximum
                
                **Try changing your settings** to see how vocabulary size affects the model's performance!
                """)

# Tab 3: Testing & Analysis
with tab3:
    st.header("🧪 Final Testing & Mistake Analysis")
    
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first!")
    else:
        splits = st.session_state.data_splits
        
        st.markdown("""
        ### 🔓 Time to Test!
        
        Now we'll evaluate our model on the **test set** - data it has never seen before.
        This gives us an honest assessment of how well our model will perform in the real world.
        """)
        
        if st.button("🧪 Evaluate on Test Set", type="primary"):
            with st.spinner("Evaluating model on test set..."):
                
                # Transform test data
                X_test_vec = st.session_state.vectorizer.transform(splits['X_test'])
                
                # Make predictions
                test_pred = st.session_state.model.predict(X_test_vec)
                test_proba = st.session_state.model.predict_proba(X_test_vec)
                
                # Calculate metrics
                metrics = calculate_comprehensive_metrics(splits['y_test'], test_pred, test_proba[:, 1])
                
                # Store results
                st.session_state.evaluation_results = {
                    'metrics': metrics,
                    'predictions': test_pred,
                    'probabilities': test_proba,
                    'true_labels': splits['y_test'],
                    'texts': splits['X_test']
                }
            
            st.success("✅ Evaluation completed!")
        
        # Display results
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            metrics = results['metrics']
            
            # Performance overview
            st.subheader("📊 Overall Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1']:.3f}")
            
            # Confusion Matrix
            st.subheader("🎯 Confusion Matrix")
            
            cm = metrics['confusion_matrix']
            cm_df = pd.DataFrame(cm, 
                               index=['Actually Negative', 'Actually Positive'],
                               columns=['Predicted Negative', 'Predicted Positive'])
            
            fig = px.imshow(cm_df, text_auto=True, aspect="auto",
                           title="Confusion Matrix: What Did We Predict vs Reality?",
                           color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curve
            col1, col2 = st.columns(2)
            
            with col1:
                if 'fpr' in metrics:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=metrics['fpr'], y=metrics['tpr'],
                        mode='lines', name=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})',
                        line=dict(color='blue', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode='lines',
                        name='Random Classifier', line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                                    yaxis_title='True Positive Rate', height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Performance interpretation
                accuracy = metrics['accuracy']
                if accuracy >= 0.9:
                    interpretation = "🌟 Excellent! This model is production-ready."
                elif accuracy >= 0.8:
                    interpretation = "👍 Good performance, could be useful with some improvements."
                elif accuracy >= 0.7:
                    interpretation = "⚠️ Moderate performance, needs significant improvement."
                else:
                    interpretation = "❌ Poor performance, back to the drawing board!"
                
                st.markdown(f"""
                ### 🎯 Performance Interpretation
                
                **{interpretation}**
                
                **What these metrics mean:**
                - **Accuracy**: {accuracy:.1%} of predictions were correct
                - **Precision**: {metrics['precision']:.1%} of positive predictions were actually positive
                - **Recall**: {metrics['recall']:.1%} of actual positive examples were caught
                - **F1-Score**: {metrics['f1']:.3f} (balanced measure of precision and recall)
                """)
            
            # Detailed mistake analysis
            analyze_misclassified_examples(
                np.array(results['true_labels']),
                results['predictions'],
                results['texts'],
                results['probabilities']
            )

# Tab 4: Try It Yourself
with tab4:
    st.header("🔮 Try the Model Yourself")
    
    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first!")
    else:
        st.markdown("Test the model with your own presentation feedback!")
        
        # Sample feedback for testing
        sample_feedback = [
            "This presentation was incredibly clear and well-structured, I learned so much!",
            "The audio quality was terrible and I couldn't understand half of what was said",
            "Great examples that really helped explain the concepts",
            "Too many technical terms without proper explanations",
            "Perfect pacing, not too fast or too slow",
            "The presenter seemed unprepared and kept making mistakes"
        ]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🎲 Try Random Sample"):
                st.session_state.sample_feedback = random.choice(sample_feedback)
            
            user_input = st.text_area(
                "Enter presentation feedback:",
                value=st.session_state.get('sample_feedback', ''),
                height=100,
                placeholder="Type your feedback here or click 'Try Random Sample'"
            )
            
            if st.button("🔍 Analyze Feedback") and user_input:
                
                # Make prediction
                user_vec = st.session_state.vectorizer.transform([user_input])
                prediction = st.session_state.model.predict(user_vec)[0]
                probabilities = st.session_state.model.predict_proba(user_vec)[0]
                confidence = max(probabilities) * 100
                
                # Display result
                sentiment = "Positive" if prediction == 1 else "Negative"
                color = "green" if prediction == 1 else "red"
                
                st.markdown(f"""
                ### 📊 Analysis Result
                
                <div style="border: 2px solid {color}; border-radius: 10px; padding: 20px; background-color: rgba({'0,255,0' if color=='green' else '255,0,0'}, 0.1);">
                    <h4>"{user_input}"</h4>
                    <p><strong>Predicted Sentiment:</strong> <span style="color:{color}; font-weight:bold;">{sentiment}</span></p>
                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <p><strong>Positive Probability:</strong> {probabilities[1]*100:.1f}%</p>
                    <p><strong>Negative Probability:</strong> {probabilities[0]*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show most influential words
                feature_names = st.session_state.vectorizer.get_feature_names_out()
                coefficients = st.session_state.model.coef_[0]
                
                # Get features for this text
                user_features = st.session_state.vectorizer.transform([user_input])
                feature_indices = user_features.nonzero()[1]
                
                if len(feature_indices) > 0:
                    st.markdown("#### 🔍 Most Influential Words:")
                    
                    word_influences = []
                    for idx in feature_indices:
                        word = feature_names[idx]
                        coef = coefficients[idx]
                        influence = user_features[0, idx] * coef
                        word_influences.append((word, influence, coef))
                    
                    # Sort by absolute influence
                    word_influences.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    for word, influence, coef in word_influences[:5]:
                        direction = "👍" if influence > 0 else "👎"
                        st.markdown(f"{direction} **{word}**: {influence:.3f} (word weight: {coef:.3f})")
        
        with col2:
            st.markdown("### 💡 Understanding Predictions")
            st.markdown("""
            **Confidence Score**: How certain the model is
            
            **Probability**: Raw probability scores
            - Higher positive probability = more positive
            - Numbers always add up to 100%
            
            **Influential Words**: 
            - Words that pushed the prediction towards positive/negative
            - Based on what the model learned from training data
            
            **Try different examples** to see how the model behaves!
            """)

# Sidebar with learning summary
with st.sidebar:
    st.markdown("## 📚 Key Learning Points")
    
    st.markdown("""
    ### What You've Learned:
    
    **🎯 Data Splitting**
    - Why we need train/validation/test splits
    - How validation prevents overfitting
    - The importance of never touching test data until the end
    
    **🔍 Model Evaluation**
    - Accuracy, precision, recall, F1-score
    - Confusion matrices
    - ROC curves and AUC
    
    **🧠 Understanding Mistakes**
    - Why models make errors
    - Patterns in misclassified examples
    - Confidence vs correctness
    
    **⚖️ Real-world ML**
    - Professional evaluation practices
    - Feature importance analysis
    - Model interpretability
    """)
    
    if st.session_state.evaluation_results:
        st.markdown("### 📊 Quick Stats")
        metrics = st.session_state.evaluation_results['metrics']
        st.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
        st.metric("Test F1-Score", f"{metrics['f1']:.3f}")
        
        # Count misclassified
        y_true = np.array(st.session_state.evaluation_results['true_labels'])
        y_pred = st.session_state.evaluation_results['predictions']
        misclassified = sum(y_true != y_pred)
        st.metric("Mistakes Made", f"{misclassified}/{len(y_true)}")