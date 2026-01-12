# 데이터와 레이블을 불러오고 전처리를 수행
import string
import email
import nltk
nltk.download('stopwords')

punctuations = list(string.punctuation)
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()

# 이메일의 여러 부분을 하나의 문자열로 합한다.
def flatten_to_string(parts):
    ret = []
    if type(parts) == str:
        ret.append(parts)
    elif type(parts) == list:
        for part in parts:
            ret += flatten_to_string(part)
    elif parts.get_content_type == 'text/plain':
        ret += parts.get_payload()
    return ret

# 이메일로부터 제목과 내용 텍스트를 추출한다.
def extract_email_text(path):
    # 압력 파일로부터 하나의 이메일을 불러온다.
    with open(path, errors='ignore') as f:
        msg = email.message_from_file(f)
    if not msg:
        return ""

    # 이메일 제목을 불러온다.
    subject = msg['Subject']
    if not subject:
        subject = ""

    # 이메일 내용을 불러온다.
    body = ' '.join(m for m in flatten_to_string(msg.get_payload()) if type(m) == str)
    if not body:
        body = ""

    return subject + ' ' + body

# 이메일을 형태소 분석한다.
def load(path):
    email_text = extract_email_text(path)
    if not email_text:
        return []

    # 메시지를 토큰화한다.
    tokens = nltk.word_tokenize(email_text)

    # 토큰에서 마침표를 제거한다.
    tokens = [i.strip("".join(punctuations)) for i in tokens if i not in punctuations]

    # 자주 사용하지 않는 단어를 제거한다.
    if len(tokens) > 2:
        return [stemmer.stem(w) for w in tokens if w not in stopwords]
    return []

import os
nltk.download('punkt_tab')

DATA_DIR = './trec07p/data/'
LABELS_FILE = './trec07p/full/index'
TRAINING_SET_RATIO = 0.7

labels = {}
spam_words = set()
ham_words = set()

# Read the labels
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

# Split corpus into train and test sets
filelist = os.listdir(DATA_DIR)
X_train = filelist[:int(len(filelist)*TRAINING_SET_RATIO)]
X_test = filelist[int(len(filelist)*TRAINING_SET_RATIO):]

for filename in X_train:
        path = os.path.join(DATA_DIR, filename)
        if filename in labels:
            label = labels[filename]
            stems = load(path)
            if not stems:
                continue
            if label == 1:
                ham_words.update(stems)
            elif label == 0:
                spam_words.update(stems)
            else:
                continue

blacklist = spam_words - ham_words

from datasketch import MinHash, MinHashLSH

# 스팸 파일만 추출한 뒤 LSH 매처(matcher)에
spam_files = [x for x in X_train if labels[x] == 0]

# MinHashLSH 매처를 자카드 유사도 모드로 설정한다. 
# MinhashLSH 임계치를 0.5로, 순열 수를 128개로 설정한다.
lsh = MinHashLSH(threshold=0.5, num_perm=128)

# 스팸 메일의 MinHash를 학습할 때 사용할 LSH 매처를 전달한다.
for idx, f in enumerate(spam_files):
    minhash = MinHash(num_perm=128)
    stems = load(os.path.join(DATA_DIR, f))
    if len(stems) < 2: continue
    for s in stems:
        minhash.update(s.encode('utf-8'))
    lsh.insert(f, minhash)

def lsh_predict_label(stems):
    '''
    LSH 매처에 쿼리하는 경우의 반환 값:
        0 : 스팸으로 예측
        1 : 햄으로 예측
       -1 : 파싱 에러
    '''
    minhash = MinHash(num_perm=128)
    if len(stems) < 2:
        return -1
    for s in stems:
        minhash.update(s.encode('utf-8'))
    matches = lsh.query(minhash)
    if matches:
        return 0
    else:
        return 1

def read_email_files():
    X = []
    y = [] 
    for i in range(len(labels)):
        filename = 'inmail.' + str(i+1)
        email_str = extract_email_text(
            os.path.join(DATA_DIR, filename))
        X.append(email_str)
        y.append(labels[filename])
    return X, y

from sklearn.model_selection import train_test_split 

X, y = read_email_files()

X_train, X_test, y_train, y_test, idx_train, idx_test = \
    train_test_split(X, y, range(len(y)), 
    train_size=TRAINING_SET_RATIO, random_state=2)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Initialize the classifier and make label predictions
mnb = MultinomialNB()
mnb.fit(X_train_vector, y_train)
y_pred = mnb.predict(X_test_vector)

# Print results
print('Classification accuracy {:.1%}'.format(accuracy_score(y_test, y_pred)))