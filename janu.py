# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset (spam_dataset.csv)
df = pd.read_csv(r'd:\1\programing languages\codetech\sioaj\spam_dataset.csv')


# Step 2: Preprocessing - Convert 'spam' to 1, 'ham' to 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Step 3: Features (X) and Labels (y)
X = df['text']
y = df['label']

# Convert text data into numerical data (feature vectors)
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Predict on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")