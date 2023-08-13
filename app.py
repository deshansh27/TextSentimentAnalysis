from flask import Flask,request,render_template
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

app=Flask(__name__)

# Load the sentiment model
with open('Sentiment_Model_kaggle_logistic_regression.pkl', 'rb') as model_file:
    sentiment_model = pickle.load(model_file)

# Load the vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# # Load the preprocess function
# with open('preprocess_model.pkl', 'rb') as preprocess_file:
#     preprocess = pickle.load(preprocess_file)

def preprocess(text):
    print("Input type:", type(text))
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    print(text)
    
    # Tokenization
    tokens = word_tokenize(text)
    print("input-2:",tokens)
       
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
       
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
       
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    #stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    print("input-3",stemmed_tokens)
    rev=' '.join(stemmed_tokens)   
    return rev

@app.route('/', methods =["GET", "POST"])
def sentiment():
    if request.method=='POST':
        text=request.form.get('text')            #('') is shoul be name 

        preprocessed_text = preprocess(text)

        vectorized_text = vectorizer.transform([preprocessed_text])

        prediction = sentiment_model.predict(vectorized_text)[0]

        if prediction==0:
            sentiments='negative'
        elif prediction==1:
            sentiments='neutral'
        else:
            sentiments='positive'
        return f"The sentiment is: {sentiments}"
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)


