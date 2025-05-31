import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Enhanced ML Sentiment Analysis Demo", layout="wide")

# Enhanced dataset
def load_enhanced_dataset():
    """Load a more comprehensive training dataset"""
    data = [
        # Original presentation feedback examples
        {"text": "I love this presentation", "sentiment": 1},
        {"text": "This is really helpful", "sentiment": 1},
        {"text": "I don't understand this", "sentiment": 0},
        {"text": "This is boring", "sentiment": 0},
        {"text": "Great explanation of machine learning", "sentiment": 1},
        {"text": "Too complicated for beginners", "sentiment": 0},
        {"text": "I'm learning so much", "sentiment": 1},
        {"text": "Unclear examples", "sentiment": 0},
        {"text": "The visuals help a lot", "sentiment": 1},
        {"text": "Moving too quickly", "sentiment": 0},
        
        # Product review examples
        {"text": "Amazing product, exceeded my expectations", "sentiment": 1},
        {"text": "Terrible quality, waste of money", "sentiment": 0},
        {"text": "Good value for the price", "sentiment": 1},
        {"text": "Poor customer service experience", "sentiment": 0},
        {"text": "Fast delivery and great packaging", "sentiment": 1},
        {"text": "Product arrived damaged", "sentiment": 0},
        {"text": "Works exactly as described", "sentiment": 1},
        {"text": "Completely useless, doesn't work", "sentiment": 0},
        {"text": "Beautiful design and easy to use", "sentiment": 1},
        {"text": "Complicated setup process", "sentiment": 0},
        
        # Movie review examples
        {"text": "Fantastic film with brilliant acting", "sentiment": 1},
        {"text": "Boring plot, terrible dialogue", "sentiment": 0},
        {"text": "Inspiring story that moved me", "sentiment": 1},
        {"text": "Predictable and disappointing", "sentiment": 0},
        {"text": "Outstanding cinematography", "sentiment": 1},
        {"text": "Poor special effects", "sentiment": 0},
        {"text": "Captivating from start to finish", "sentiment": 1},
        {"text": "Couldn't wait for it to end", "sentiment": 0},
        {"text": "Excellent character development", "sentiment": 1},
        {"text": "Weak storyline and poor acting", "sentiment": 0},
        
        # Restaurant review examples
        {"text": "Delicious food and friendly staff", "sentiment": 1},
        {"text": "Overpriced and tasteless", "sentiment": 0},
        {"text": "Wonderful atmosphere and service", "sentiment": 1},
        {"text": "Long wait times and cold food", "sentiment": 0},
        {"text": "Fresh ingredients and creative dishes", "sentiment": 1},
        {"text": "Dirty tables and rude waiters", "sentiment": 0},
        {"text": "Perfect for a romantic dinner", "sentiment": 1},
        {"text": "Noisy and uncomfortable seating", "sentiment": 0},
        {"text": "Best meal I've had in years", "sentiment": 1},
        {"text": "Food poisoning from this place", "sentiment": 0},
        
        # App/Software review examples
        {"text": "User-friendly interface and smooth performance", "sentiment": 1},
        {"text": "Crashes constantly and full of bugs", "sentiment": 0},
        {"text": "Great features and regular updates", "sentiment": 1},
        {"text": "Confusing navigation and slow loading", "sentiment": 0},
        {"text": "Exactly what I needed, highly recommend", "sentiment": 1},
        {"text": "Waste of storage space", "sentiment": 0},
        {"text": "Innovative design and helpful tutorials", "sentiment": 1},
        {"text": "Lacks basic functionality", "sentiment": 0},
        {"text": "Responsive customer support", "sentiment": 1},
        {"text": "No help available when issues arise", "sentiment": 0},
        
        # Social media style examples
        {"text": "Having an amazing day with friends", "sentiment": 1},
        {"text": "Feeling frustrated and annoyed", "sentiment": 0},
        {"text": "So grateful for this opportunity", "sentiment": 1},
        {"text": "This weather is absolutely terrible", "sentiment": 0},
        {"text": "Proud of my team's achievements", "sentiment": 1},
        {"text": "Disappointed by the poor service", "sentiment": 0},
        {"text": "Excited about the weekend plans", "sentiment": 1},
        {"text": "Stressed about upcoming deadlines", "sentiment": 0},
        {"text": "Love spending time with family", "sentiment": 1},
        {"text": "Hate dealing with bureaucracy", "sentiment": 0},
        
        # Educational examples
        {"text": "Brilliant teacher who explains clearly", "sentiment": 1},
        {"text": "Confusing lectures and poor organisation", "sentiment": 0},
        {"text": "Course material is engaging and relevant", "sentiment": 1},
        {"text": "Outdated content and boring assignments", "sentiment": 0},
        {"text": "Helpful feedback and fair grading", "sentiment": 1},
        {"text": "Unreasonable expectations and harsh marking", "sentiment": 0},
        {"text": "Interactive sessions that enhance learning", "sentiment": 1},
        {"text": "Passive teaching style puts me to sleep", "sentiment": 0},
        {"text": "Well-structured curriculum", "sentiment": 1},
        {"text": "Disorganised and poorly planned", "sentiment": 0},
        
        # Travel examples
        {"text": "Breathtaking views and comfortable hotel", "sentiment": 1},
        {"text": "Tourist trap with overpriced everything", "sentiment": 0},
        {"text": "Memorable experiences and friendly locals", "sentiment": 1},
        {"text": "Disappointing destination, not worth the trip", "sentiment": 0},
        {"text": "Perfect weather and stunning scenery", "sentiment": 1},
        {"text": "Crowded beaches and poor accommodations", "sentiment": 0},
        {"text": "Rich culture and fascinating history", "sentiment": 1},
        {"text": "Language barrier made everything difficult", "sentiment": 0},
        {"text": "Adventure of a lifetime", "sentiment": 1},
        {"text": "Stressful travel with multiple delays", "sentiment": 0},
    ]
    return data

# Custom Model (Enhanced version of original)
class CustomSentimentModel:
    def __init__(self):
        self.training_data = load_enhanced_dataset()
        self.vocabulary = {}
        self.word_weights = {'__bias__': 0}
        self.trained = False
        self.validation_scores = []
        
    def create_vocabulary(self):
        """Enhanced vocabulary creation with better preprocessing"""
        word_freq = {}
        bigram_freq = {}
        
        for item in self.training_data:
            # Better text preprocessing
            text = re.sub(r'[^\w\s]', '', item["text"].lower())
            words = [word for word in text.split() if len(word) > 2]  # Filter short words
            
            # Count individual words
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Count bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]}_{words[i+1]}"
                bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
        
        # Filter based on frequency
        min_frequency = 3
        self.vocabulary = {}
        
        for word, freq in word_freq.items():
            if freq >= min_frequency:
                self.vocabulary[word] = True
                
        for bigram, freq in bigram_freq.items():
            if freq >= min_frequency:
                self.vocabulary[bigram] = True
        
        # Add special features
        self.vocabulary['__has_negation__'] = True
        self.vocabulary['__exclamation__'] = True
        self.vocabulary['__question__'] = True
        
    def extract_features(self, text):
        """Enhanced feature extraction"""
        features = {}
        
        # Normalise text
        text_clean = re.sub(r'[^\w\s!?]', '', text.lower())
        words = [word for word in text_clean.split() if len(word) > 2]
        
        # Unigrams
        for word in words:
            if word in self.vocabulary:
                features[word] = features.get(word, 0) + 1
        
        # Bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            if bigram in self.vocabulary:
                features[bigram] = features.get(bigram, 0) + 1
        
        # Enhanced special features
        negation_words = ['not', 'no', "don't", "didn't", "doesn't", "isn't", "aren't", "wasn't", "weren't", 'never', 'nothing', 'nobody', 'nowhere']
        if any(word in negation_words for word in words):
            features['__has_negation__'] = 1
            
        if '!' in text:
            features['__exclamation__'] = 1
            
        if '?' in text:
            features['__question__'] = 1
            
        return features
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
    
    def predict_proba(self, text):
        """Return probability for compatibility with sklearn"""
        features = self.extract_features(text)
        score = self.word_weights.get('__bias__', 0)
        
        for word, count in features.items():
            if word in self.word_weights:
                score += count * self.word_weights[word]
        
        prob_positive = self.sigmoid(score)
        return np.array([1 - prob_positive, prob_positive])
    
    def predict(self, texts):
        """Predict for multiple texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = []
        for text in texts:
            prob = self.predict_proba(text)[1]
            predictions.append(1 if prob > 0.5 else 0)
        
        return np.array(predictions)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=30, callback=None):
        """Enhanced training with validation tracking"""
        self.training_data = [{"text": text, "sentiment": label} for text, label in zip(X_train, y_train)]
        self.create_vocabulary()
        
        # Initialise weights
        self.word_weights = {'__bias__': 0}
        for word in self.vocabulary:
            self.word_weights[word] = np.random.uniform(-0.1, 0.1)
        
        history = []
        base_learning_rate = 0.1
        
        for epoch in range(epochs):
            # Adjust learning rate
            if epoch < epochs // 3:
                learning_rate = base_learning_rate
            elif epoch < 2 * epochs // 3:
                learning_rate = base_learning_rate / 2
            else:
                learning_rate = base_learning_rate / 4
            
            # Training
            total_loss = 0
            indices = np.random.permutation(len(X_train))
            
            for idx in indices:
                features = self.extract_features(X_train[idx])
                prediction = self.predict_proba(X_train[idx])[1]
                target = y_train[idx]
                error = target - prediction
                
                # Update weights
                self.word_weights['__bias__'] += learning_rate * error
                for word, count in features.items():
                    self.word_weights[word] = self.word_weights.get(word, 0) + learning_rate * error * count
                
                total_loss += error * error
            
            # Regularisation
            regularisation_rate = 0.01
            for word in self.word_weights:
                if word != '__bias__':
                    self.word_weights[word] *= (1 - regularisation_rate * learning_rate)
            
            # Validation
            val_predictions = self.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            
            # Calculate additional metrics
            val_precision = precision_score(y_val, val_predictions, zero_division=0)
            val_recall = recall_score(y_val, val_predictions, zero_division=0)
            val_f1 = f1_score(y_val, val_predictions, zero_division=0)
            
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': total_loss / len(X_train),
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1
            }
            
            history.append(epoch_metrics)
            
            if callback:
                callback(epoch + 1, epochs, epoch_metrics)
            
            time.sleep(0.05)  # Visual delay
        
        self.trained = True
        return history

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

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Create an enhanced confusion matrix plot"""
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Negative', 'Positive'],
                    y=['Negative', 'Positive'],
                    title=title,
                    color_continuous_scale='Blues',
                    text_auto=True)
    
    fig.update_layout(width=400, height=300)
    return fig

def plot_metrics_comparison(custom_metrics, sklearn_metrics):
    """Create a comparison chart of model metrics"""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    custom_values = [custom_metrics['accuracy'], custom_metrics['precision'], 
                    custom_metrics['recall'], custom_metrics['f1']]
    sklearn_values = [sklearn_metrics['accuracy'], sklearn_metrics['precision'],
                     sklearn_metrics['recall'], sklearn_metrics['f1']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Custom Model',
        x=metrics_names,
        y=custom_values,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Scikit-Learn Model',
        x=metrics_names,
        y=sklearn_values,
        marker_color='orange'
    ))
    
    fig.update_layout(
        title='Model Comparison: Custom vs Scikit-Learn',
        yaxis_title='Score',
        xaxis_title='Metrics',
        barmode='group',
        width=600, height=400
    )
    
    return fig

# Initialise session state
if 'custom_model' not in st.session_state:
    st.session_state.custom_model = CustomSentimentModel()
if 'sklearn_model' not in st.session_state:
    st.session_state.sklearn_model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'data_splits' not in st.session_state:
    st.session_state.data_splits = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'test_predictions' not in st.session_state:
    st.session_state.test_predictions = []

# App title and description
st.title("🧠 Enhanced ML Sentiment Analysis Demo")

st.markdown("""
This advanced demo compares two approaches to sentiment analysis:
- **Custom Model**: Educational implementation showing ML fundamentals
- **Scikit-Learn Model**: Professional-grade machine learning

Key learning concepts: **Train/Validation/Test splits**, **Precision vs Recall trade-offs**, **ROC curves**, **Model comparison**
""")

# Create data splits
@st.cache_data
def create_data_splits():
    """Create train/validation/test splits from the enhanced dataset"""
    data = load_enhanced_dataset()
    
    texts = [item['text'] for item in data]
    labels = [item['sentiment'] for item in data]
    
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
        'X_test': X_test, 'y_test': y_test
    }

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data & Training", "🔮 Prediction", "📈 Model Evaluation", "🎯 Advanced Analysis"])

# Tab 1: Data & Training
with tab1:
    st.header("Dataset and Model Training")
    
    # Show data splits
    if st.session_state.data_splits is None:
        st.session_state.data_splits = create_data_splits()
    
    splits = st.session_state.data_splits
    
    # Display dataset information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Set", f"{len(splits['X_train'])} examples")
    with col2:
        st.metric("Validation Set", f"{len(splits['X_val'])} examples")
    with col3:
        st.metric("Test Set", f"{len(splits['X_test'])} examples")
    
    # Show sample data
    with st.expander("📝 View Sample Training Data"):
        sample_df = pd.DataFrame({
            'Text': splits['X_train'][:10],
            'Sentiment': ['Positive' if label == 1 else 'Negative' for label in splits['y_train'][:10]]
        })
        st.dataframe(sample_df, use_container_width=True)
    
    st.markdown("---")
    
    # Training section
    st.subheader("🏋️ Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Parameters:**")
        epochs = st.slider("Training Epochs:", 10, 50, 20)
        
        if st.button("🚀 Train Both Models", type="primary"):
            
            with st.spinner("Training models..."):
                # Train custom model
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def training_callback(epoch, total_epochs, metrics):
                    progress = epoch / total_epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Training Custom Model - Epoch {epoch}/{total_epochs} - Val Accuracy: {metrics['val_accuracy']*100:.1f}%")
                
                # Train custom model
                history = st.session_state.custom_model.train(
                    splits['X_train'], splits['y_train'],
                    splits['X_val'], splits['y_val'],
                    epochs=epochs, callback=training_callback
                )
                st.session_state.training_history = history
                
                # Train scikit-learn model
                status_text.text("Training Scikit-Learn Model...")
                
                # Create TF-IDF vectoriser
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
                X_train_vec = vectorizer.fit_transform(splits['X_train'])
                X_val_vec = vectorizer.transform(splits['X_val'])
                
                # Train logistic regression
                sklearn_model = LogisticRegression(random_state=42, max_iter=1000)
                sklearn_model.fit(X_train_vec, splits['y_train'])
                
                st.session_state.sklearn_model = sklearn_model
                st.session_state.vectorizer = vectorizer
                
                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                
            st.success("✅ Both models trained successfully!")
    
    with col2:
        # Training progress visualisation
        if st.session_state.training_history:
            st.markdown("**Custom Model Training Progress:**")
            
            history_df = pd.DataFrame(st.session_state.training_history)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Validation Accuracy', 'Training Loss', 'Precision', 'Recall'),
                vertical_spacing=0.1
            )
            
            # Accuracy
            fig.add_trace(go.Scatter(x=history_df['epoch'], y=history_df['val_accuracy'], 
                                   name='Val Accuracy', line=dict(color='blue')), row=1, col=1)
            
            # Loss
            fig.add_trace(go.Scatter(x=history_df['epoch'], y=history_df['train_loss'], 
                                   name='Train Loss', line=dict(color='red')), row=1, col=2)
            
            # Precision
            fig.add_trace(go.Scatter(x=history_df['epoch'], y=history_df['val_precision'], 
                                   name='Val Precision', line=dict(color='green')), row=2, col=1)
            
            # Recall
            fig.add_trace(go.Scatter(x=history_df['epoch'], y=history_df['val_recall'], 
                                   name='Val Recall', line=dict(color='orange')), row=2, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Prediction
with tab2:
    st.header("🔮 Model Predictions")
    
    if st.session_state.custom_model.trained and st.session_state.sklearn_model is not None:
        
        # Sample texts for testing
        sample_texts = [
            "This movie was absolutely fantastic! Amazing acting and brilliant story.",
            "Terrible product, completely useless and overpriced garbage.",
            "The restaurant had okay food but service was disappointing.",
            "Love this app! So user-friendly and helpful for daily tasks.",
            "Boring presentation with unclear examples and poor organisation."
        ]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            if st.button("🎲 Try Random Sample"):
                st.session_state.sample_text = random.choice(sample_texts)
            
            user_input = st.text_area(
                "Enter text to analyse:",
                value=st.session_state.get('sample_text', ''),
                height=100,
                placeholder="Type your text here or click 'Try Random Sample'"
            )
            
            if st.button("🔍 Analyze Text") and user_input:
                
                # Custom model prediction
                custom_proba = st.session_state.custom_model.predict_proba(user_input)
                custom_pred = 1 if custom_proba[1] > 0.5 else 0
                custom_confidence = max(custom_proba) * 100
                
                # Scikit-learn prediction
                user_vec = st.session_state.vectorizer.transform([user_input])
                sklearn_proba = st.session_state.sklearn_model.predict_proba(user_vec)[0]
                sklearn_pred = st.session_state.sklearn_model.predict(user_vec)[0]
                sklearn_confidence = max(sklearn_proba) * 100
                
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                # Create comparison
                results_df = pd.DataFrame({
                    'Model': ['Custom Model', 'Scikit-Learn'],
                    'Prediction': [
                        'Positive' if custom_pred == 1 else 'Negative',
                        'Positive' if sklearn_pred == 1 else 'Negative'
                    ],
                    'Confidence': [f"{custom_confidence:.1f}%", f"{sklearn_confidence:.1f}%"],
                    'Positive Probability': [f"{custom_proba[1]:.3f}", f"{sklearn_proba[1]:.3f}"]
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Probability comparison chart
                fig = go.Figure()
                
                models = ['Custom Model', 'Scikit-Learn Model']
                positive_probs = [custom_proba[1], sklearn_proba[1]]
                negative_probs = [custom_proba[0], sklearn_proba[0]]
                
                fig.add_trace(go.Bar(name='Negative', x=models, y=negative_probs, marker_color='lightcoral'))
                fig.add_trace(go.Bar(name='Positive', x=models, y=positive_probs, marker_color='lightgreen'))
                
                fig.update_layout(
                    title='Prediction Probability Comparison',
                    yaxis_title='Probability',
                    barmode='stack',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Store prediction for evaluation
                prediction_result = {
                    'text': user_input,
                    'custom_pred': custom_pred,
                    'sklearn_pred': sklearn_pred,
                    'custom_proba': custom_proba[1],
                    'sklearn_proba': sklearn_proba[1],
                    'timestamp': time.time()
                }
                st.session_state.test_predictions.append(prediction_result)
        
        with col2:
            st.markdown("### 💡 Understanding the Results")
            st.markdown("""
            **Confidence Score**: How certain the model is about its prediction
            
            **Probability**: Raw probability of positive sentiment (0-1)
            
            **Model Differences**: 
            - Custom model uses simple word weights
            - Scikit-Learn uses TF-IDF + advanced optimisation
            
            **Why might they differ?**
            - Different feature extraction methods
            - Different training algorithms
            - Different regularisation techniques
            """)
    else:
        st.warning("⚠️ Please train the models first in the 'Data & Training' tab!")

# Tab 3: Model Evaluation
with tab3:
    st.header("📈 Model Performance Evaluation")
    
    if st.session_state.custom_model.trained and st.session_state.sklearn_model is not None:
        
        splits = st.session_state.data_splits
        
        # Evaluate both models on test set
        if st.button("🧪 Evaluate on Test Set"):
            
            with st.spinner("Evaluating models on test set..."):
                
                # Custom model evaluation
                custom_test_pred = st.session_state.custom_model.predict(splits['X_test'])
                custom_test_proba = [st.session_state.custom_model.predict_proba(text)[1] for text in splits['X_test']]
                custom_metrics = calculate_comprehensive_metrics(splits['y_test'], custom_test_pred, custom_test_proba)
                
                # Scikit-learn evaluation
                X_test_vec = st.session_state.vectorizer.transform(splits['X_test'])
                sklearn_test_pred = st.session_state.sklearn_model.predict(X_test_vec)
                sklearn_test_proba = st.session_state.sklearn_model.predict_proba(X_test_vec)[:, 1]
                sklearn_metrics = calculate_comprehensive_metrics(splits['y_test'], sklearn_test_pred, sklearn_test_proba)
                
                st.session_state.evaluation_results = {
                    'custom': custom_metrics,
                    'sklearn': sklearn_metrics
                }
            
            st.success("✅ Evaluation completed!")
        
        # Display evaluation results
        if st.session_state.evaluation_results:
            
            custom_metrics = st.session_state.evaluation_results['custom']
            sklearn_metrics = st.session_state.evaluation_results['sklearn']
            
            # Metrics comparison
            st.subheader("📊 Performance Metrics Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Custom Model:**")
                st.metric("Accuracy", f"{custom_metrics['accuracy']:.3f}")
                st.metric("Precision", f"{custom_metrics['precision']:.3f}")
                st.metric("Recall", f"{custom_metrics['recall']:.3f}")
                st.metric("F1-Score", f"{custom_metrics['f1']:.3f}")
                if 'roc_auc' in custom_metrics:
                    st.metric("ROC AUC", f"{custom_metrics['roc_auc']:.3f}")
            
            with col2:
                st.markdown("**Scikit-Learn Model:**")
                st.metric("Accuracy", f"{sklearn_metrics['accuracy']:.3f}")
                st.metric("Precision", f"{sklearn_metrics['precision']:.3f}")
                st.metric("Recall", f"{sklearn_metrics['recall']:.3f}")
                st.metric("F1-Score", f"{sklearn_metrics['f1']:.3f}")
                if 'roc_auc' in sklearn_metrics:
                    st.metric("ROC AUC", f"{sklearn_metrics['roc_auc']:.3f}")
            
            # Comparison chart
            comparison_fig = plot_metrics_comparison(custom_metrics, sklearn_metrics)
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Confusion matrices
            st.subheader("🎯 Confusion Matrices")
            
            col1, col2 = st.columns(2)
            
            with col1:
                custom_cm_fig = plot_confusion_matrix(custom_metrics['confusion_matrix'], "Custom Model")
                st.plotly_chart(custom_cm_fig, use_container_width=True)
            
            with col2:
                sklearn_cm_fig = plot_confusion_matrix(sklearn_metrics['confusion_matrix'], "Scikit-Learn Model")
                st.plotly_chart(sklearn_cm_fig, use_container_width=True)
            
            # ROC Curves
            if 'fpr' in custom_metrics and 'fpr' in sklearn_metrics:
                st.subheader("📈 ROC Curves")
                
                fig = go.Figure()
                
                # Custom model ROC
                fig.add_trace(go.Scatter(
                    x=custom_metrics['fpr'], 
                    y=custom_metrics['tpr'],
                    mode='lines',
                    name=f'Custom Model (AUC = {custom_metrics["roc_auc"]:.3f})',
                    line=dict(color='blue', width=3)
                ))
                
                # Scikit-learn ROC
                fig.add_trace(go.Scatter(
                    x=sklearn_metrics['fpr'], 
                    y=sklearn_metrics['tpr'],
                    mode='lines',
                    name=f'Scikit-Learn Model (AUC = {sklearn_metrics["roc_auc"]:.3f})',
                    line=dict(color='orange', width=3)
                ))
                
                # Random classifier line
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                fig.update_layout(
                    title='ROC Curve Comparison',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation
                with st.expander("🎓 Understanding ROC Curves"):
                    st.markdown("""
                    **ROC (Receiver Operating Characteristic) Curves** show the trade-off between:
                    - **True Positive Rate (Sensitivity)**: How well the model catches actual positives
                    - **False Positive Rate**: How often the model incorrectly predicts positive
                    
                    **Perfect Model**: Would hug the top-left corner (100% true positives, 0% false positives)
                    
                    **Random Model**: Diagonal line (no better than coin flipping)
                    
                    **AUC (Area Under Curve)**: Single number summary
                    - 1.0 = Perfect model
                    - 0.5 = Random guessing
                    - Higher is better
                    """)
    else:
        st.warning("⚠️ Please train the models first!")

# Tab 4: Advanced Analysis
with tab4:
    st.header("🎯 Advanced Analysis & Insights")
    
    if st.session_state.evaluation_results:
        
        # Precision-Recall Trade-off Analysis
        st.subheader("⚖️ Precision vs Recall Trade-off")
        
        st.markdown("""
        **The Fundamental Trade-off**: You can't usually maximise both precision and recall simultaneously!
        
        - **High Precision Strategy**: Be conservative, only predict positive when very confident
        - **High Recall Strategy**: Be aggressive, catch all possible positives
        """)
        
        # Interactive threshold analysis
        threshold = st.slider("🎚️ Decision Threshold", 0.1, 0.9, 0.5, 0.01)
        
        if st.session_state.sklearn_model and st.session_state.evaluation_results:
            
            # Recalculate predictions with new threshold
            splits = st.session_state.data_splits
            X_test_vec = st.session_state.vectorizer.transform(splits['X_test'])
            test_proba = st.session_state.sklearn_model.predict_proba(X_test_vec)[:, 1]
            test_pred_new = (test_proba >= threshold).astype(int)
            
            # Calculate metrics with new threshold
            new_precision = precision_score(splits['y_test'], test_pred_new, zero_division=0)
            new_recall = recall_score(splits['y_test'], test_pred_new, zero_division=0)
            new_f1 = f1_score(splits['y_test'], test_pred_new, zero_division=0)
            new_accuracy = accuracy_score(splits['y_test'], test_pred_new)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precision", f"{new_precision:.3f}")
            with col2:
                st.metric("Recall", f"{new_recall:.3f}")
            with col3:
                st.metric("F1-Score", f"{new_f1:.3f}")
            with col4:
                st.metric("Accuracy", f"{new_accuracy:.3f}")
        
        # Feature importance analysis
        st.subheader("🔍 Feature Importance Analysis")
        
        if st.session_state.sklearn_model and st.session_state.vectorizer:
            
            # Get feature importance from logistic regression
            feature_names = st.session_state.vectorizer.get_feature_names_out()
            coefficients = st.session_state.sklearn_model.coef_[0]
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Most Positive Features:**")
                positive_features = feature_importance[feature_importance['coefficient'] > 0].head(10)
                for _, row in positive_features.iterrows():
                    st.markdown(f"- **{row['feature']}**: +{row['coefficient']:.3f}")
            
            with col2:
                st.markdown("**Most Negative Features:**")
                negative_features = feature_importance[feature_importance['coefficient'] < 0].head(10)
                for _, row in negative_features.iterrows():
                    st.markdown(f"- **{row['feature']}**: {row['coefficient']:.3f}")
            
            # Feature importance chart
            top_features = pd.concat([
                feature_importance[feature_importance['coefficient'] > 0].head(10),
                feature_importance[feature_importance['coefficient'] < 0].head(10)
            ]).sort_values('coefficient')
            
            fig = go.Figure(go.Bar(
                x=top_features['coefficient'],
                y=top_features['feature'],
                orientation='h',
                marker_color=['red' if x < 0 else 'green' for x in top_features['coefficient']]
            ))
            
            fig.update_layout(
                title='Top 20 Most Important Features',
                xaxis_title='Coefficient Value',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison insights
        st.subheader("🤔 Model Comparison Insights")
        
        custom_metrics = st.session_state.evaluation_results['custom']
        sklearn_metrics = st.session_state.evaluation_results['sklearn']
        
        # Determine which model is better for different metrics
        better_accuracy = "Scikit-Learn" if sklearn_metrics['accuracy'] > custom_metrics['accuracy'] else "Custom"
        better_precision = "Scikit-Learn" if sklearn_metrics['precision'] > custom_metrics['precision'] else "Custom"
        better_recall = "Scikit-Learn" if sklearn_metrics['recall'] > custom_metrics['recall'] else "Custom"
        better_f1 = "Scikit-Learn" if sklearn_metrics['f1'] > custom_metrics['f1'] else "Custom"
        
        st.markdown(f"""
        ### 📈 Performance Summary:
        
        - **Best Accuracy**: {better_accuracy} Model ({max(sklearn_metrics['accuracy'], custom_metrics['accuracy']):.3f})
        - **Best Precision**: {better_precision} Model ({max(sklearn_metrics['precision'], custom_metrics['precision']):.3f})
        - **Best Recall**: {better_recall} Model ({max(sklearn_metrics['recall'], custom_metrics['recall']):.3f})
        - **Best F1-Score**: {better_f1} Model ({max(sklearn_metrics['f1'], custom_metrics['f1']):.3f})
        
        ### 🎓 Key Learning Points:
        
        **Why Scikit-Learn Usually Performs Better:**
        - Uses TF-IDF weighting (considers word importance across documents)
        - Advanced optimisation algorithms (LBFGS, SAG)
        - Built-in regularisation techniques
        - Years of optimisation and testing
        
        **Value of the Custom Model:**
        - Transparency: You can see exactly how it works
        - Educational: Demonstrates core ML principles
        - Interpretability: Easy to understand feature weights
        - Customisation: Can be modified for specific needs
        
        **When to Use Each:**
        - **Custom Model**: Learning, experimentation, full control needed
        - **Scikit-Learn**: Production systems, best performance, time constraints
        """)
        
        # Practical recommendations
        with st.expander("💡 Practical Recommendations"):
            st.markdown("""
            ### For Different Use Cases:
            
            **📧 Email Spam Detection:**
            - Prioritise **high precision** (few false positives)
            - Better to miss some spam than block important emails
            - Threshold: ~0.7-0.8
            
            **🏥 Medical Screening:**
            - Prioritise **high recall** (catch all potential cases)
            - Better to have false alarms than miss diagnoses
            - Threshold: ~0.2-0.3
            
            **📱 Content Recommendation:**
            - Balance precision and recall (**F1-score**)
            - Want good recommendations without too much noise
            - Threshold: ~0.5
            
            **💰 Fraud Detection:**
            - High precision to avoid blocking legitimate transactions
            - But also need reasonable recall to catch fraud
            - Often use ensemble methods and multiple thresholds
            """)
    
    else:
        st.warning("⚠️ Please run model evaluation first!")

# Sidebar with additional information
with st.sidebar:
    st.markdown("## 📚 Learning Resources")
    
    st.markdown("""
    ### Key Concepts Covered:
    - **Train/Validation/Test Splits**
    - **Precision vs Recall Trade-offs**
    - **ROC Curves & AUC**
    - **Feature Importance**
    - **Model Comparison**
    - **TF-IDF Vectorisation**
    
    ### Next Steps:
    1. Try different thresholds
    2. Compare feature importance
    3. Test edge cases
    4. Understand when each model fails
    """)
    
    if st.session_state.evaluation_results:
        st.markdown("### 📊 Quick Stats")
        custom_acc = st.session_state.evaluation_results['custom']['accuracy']
        sklearn_acc = st.session_state.evaluation_results['sklearn']['accuracy']
        
        st.metric("Custom Model Accuracy", f"{custom_acc:.1%}")
        st.metric("Scikit-Learn Accuracy", f"{sklearn_acc:.1%}")
        
        improvement = ((sklearn_acc - custom_acc) / custom_acc * 100) if custom_acc > 0 else 0
        st.metric("Performance Improvement", f"{improvement:.1f}%")