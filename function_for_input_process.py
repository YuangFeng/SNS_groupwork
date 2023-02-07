import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize

#If an NPL is required, a dataset of questions will be included. 
# Otherwise, we can just use the token to generate a response.
def tokenize_text(input_text):
    tokens = word_tokenize(input_text)
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return ' '.join(tokens)


def classify_intent(text):
    if 'hello' in text:
        return 'greeting'
    elif 'bye' in text:
        return 'farewell'
    else:
        return 'none'


def generate_response(intent):
    if intent == 'greeting':
        return 'Hello! How can I help you today?'
    elif intent == 'farewell':
        return 'Goodbye! Have a great day.'
    else:
        return 'I am not sure what you mean.'


def chatbot_response(input_text):
    processed_text = tokenize_text(input_text)
    intent = classify_intent(processed_text)
    response = generate_response(intent)
    return response


print(chatbot_response("HELLO from Haofan"))