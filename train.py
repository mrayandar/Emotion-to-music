import pandas as pd 
import re  
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import classification_report, accuracy_score  
import joblib  

print("Loading dataset...")  
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['sentiment', 'text']

# 2. Filter for Happy and Sad tweets
# Keep only tweets with sentiment 0 (Negative/Sad) or 4 (Positive/Happy)
df = df[df['sentiment'].isin([0, 4])]
# Create a new 'emotion' column mapping numerical sentiments to descriptive labels
# 0 -> 'Sad', 4 -> 'Happy' for binary classification
df['emotion'] = df['sentiment'].map({0: 'Sad', 4: 'Happy'})

def preprocess(text):

    text = text.lower()  # Convert to lowercase for consistency
    text = re.sub(r"http\S+", "", text)  # Remove URLs (e.g., http://example.com)
    text = re.sub(r"@\w+", "", text)  # Remove mentions (e.g., @username)
    text = re.sub(r"#\w+", "", text)  # Remove hashtags (e.g., #happy)
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation and special characters
    text = re.sub(r"\d+", "", text)  # Remove digits (e.g., 123)
    text = text.strip()  # Remove leading/trailing whitespace
    return text

df['clean_text'] = df['text'].apply(preprocess)

print("Vectorizing text...")
# Initialize TF-IDF Vectorizer with a maximum of 5,000 features to limit vocabulary size
vectorizer = TfidfVectorizer(max_features=5000)
# Fit and transform the cleaned text into TF-IDF features
X = vectorizer.fit_transform(df['clean_text'])
# Extract the target variable (emotion labels: Happy or Sad)
y = df['emotion']

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the logistic regression model
print("Training model...") 
model = LogisticRegression(max_iter=200)
# Train the model on the training data (TF-IDF features and emotion labels)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
# Print the accuracy score (proportion of correct predictions)
print("Accuracy:", accuracy_score(y_test, y_pred))
# Print detailed evaluation metrics (precision, recall, F1-score) for both classes
print("Classification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, 'emotion_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer saved.")  # Confirm successful saving