import spacy

nlp = spacy.load("en_core_web_sm")
FILLER_WORDS = {"um","uh","ah","hmm","like","so","you know","actually","just",'right','ok'}  # keep small list

def preprocess_text(text):
    doc = nlp(text.lower())
    cleaned_sentences = []

    for sent in doc.sents:
        words = [token.text for token in sent if token.text.lower() not in FILLER_WORDS]
        sentence = " ".join(words).strip()
        if len(sentence.split()) > 2:  # skip tiny fragments
            cleaned_sentences.append(sentence)
    return cleaned_sentences
