import re
import emoji

def preprocess_text(text):

    # Convert to string
    text = str(text)

    # Normalize punctuation
    text = re.sub(r'([!?.,])\1+', r'\1', text)

    # Normalize spacing
    text = re.sub(r'\s+', ' ', text)

    # Remove hidden unicode
    text = text.encode('utf-8', 'ignore').decode('utf-8')

    # Convert emojis
    text = emoji.demojize(text, delimiters=(" ", " "))

    return text.strip()
