import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV


def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def DDM(user_query: str) -> list:
    processed = preprocess(user_query)
    X = vectorizer.transform([processed])
    pred = model.predict(X)
    return list(mlb.inverse_transform(pred)[0])


with open("training_data.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

train_queries = [preprocess(item["query"]) for item in train_data]
train_labels = [item["labels"] for item in train_data]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
X_train = vectorizer.fit_transform(train_queries)

mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_labels)

model = MultiOutputClassifier(
    CalibratedClassifierCV(LinearSVC(max_iter=2000, class_weight="balanced"))
)
model.fit(X_train, Y_train)


with open("test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

extra_test_data = [
    {"query": "who was alexander the great",                    "labels": ["general"]},
    {"query": "what is the speed of sound",                     "labels": ["general"]},
    {"query": "explain the water cycle",                        "labels": ["general"]},
    {"query": "what is photosynthesis",                         "labels": ["general"]},
    {"query": "who invented the internet",                      "labels": ["general"]},
    {"query": "what is quantum mechanics",                      "labels": ["general"]},
    {"query": "what is the boiling point of water",             "labels": ["general"]},
    {"query": "who was julius caesar",                          "labels": ["general"]},
    {"query": "what is the theory of evolution",                "labels": ["general"]},
    {"query": "how does a black hole form",                     "labels": ["general"]},
    {"query": "write a blog post about climate change",         "labels": ["content"]},
    {"query": "write an email to my manager about leave",       "labels": ["content"]},
    {"query": "write a linkedin post about my new job",         "labels": ["content"]},
    {"query": "write a cover letter for a data analyst role",   "labels": ["content"]},
    {"query": "write a product description for a smartwatch",   "labels": ["content"]},
    {"query": "write an instagram caption for my photo",        "labels": ["content"]},
    {"query": "write a short story about a time traveler",      "labels": ["content"]},
    {"query": "draft a resignation letter",                     "labels": ["content"]},
    {"query": "write a thank you email after an interview",     "labels": ["content"]},
    {"query": "write a birthday message for my friend",         "labels": ["content"]},
    {"query": "generate image of a sunset over the ocean",      "labels": ["generate image"]},
    {"query": "generate image of a futuristic city at night",   "labels": ["generate image"]},
    {"query": "generate image of a wolf in a snowy forest",     "labels": ["generate image"]},
    {"query": "generate image of an astronaut on the moon",     "labels": ["generate image"]},
    {"query": "generate image of a dragon breathing fire",      "labels": ["generate image"]},
    {"query": "generate image of a cozy coffee shop interior",  "labels": ["generate image"]},
    {"query": "generate image of a samurai standing in rain",   "labels": ["generate image"]},
    {"query": "generate image of a cute baby panda",            "labels": ["generate image"]},
    {"query": "generate image of a medieval knight",            "labels": ["generate image"]},
    {"query": "generate image of a cyberpunk street",           "labels": ["generate image"]},
    {"query": "search best python courses on google",           "labels": ["google search"]},
    {"query": "search how to get a uk visa on google",          "labels": ["google search"]},
    {"query": "search top 10 movies of 2024 on google",         "labels": ["google search"]},
    {"query": "search best hospitals in lahore on google",      "labels": ["google search"]},
    {"query": "search how to lose belly fat on google",         "labels": ["google search"]},
    {"query": "search elon musk net worth on google",           "labels": ["google search"]},
    {"query": "search freelancing platforms on google",         "labels": ["google search"]},
    {"query": "search cheapest flights to dubai on google",     "labels": ["google search"]},
    {"query": "search how to learn arabic on google",           "labels": ["google search"]},
    {"query": "search latest mobile phones on google",          "labels": ["google search"]},
    {"query": "remind me to call mom at 6pm",                   "labels": ["reminder"]},
    {"query": "set a reminder for my meeting at 10am tomorrow", "labels": ["reminder"]},
    {"query": "remind me to drink water every hour",            "labels": ["reminder"]},
    {"query": "set an alarm for 5:30am",                        "labels": ["reminder"]},
    {"query": "remind me to submit the assignment by friday",   "labels": ["reminder"]},
    {"query": "remind me to pay my internet bill",              "labels": ["reminder"]},
    {"query": "set a reminder for dad's birthday next week",    "labels": ["reminder"]},
    {"query": "remind me to take my medicine at 9pm",           "labels": ["reminder"]},
    {"query": "set a reminder to exercise every morning at 7",  "labels": ["reminder"]},
    {"query": "remind me to review my notes tonight",           "labels": ["reminder"]},
    {"query": "mute the volume",                                "labels": ["system"]},
    {"query": "increase the brightness",                        "labels": ["system"]},
    {"query": "take a screenshot",                              "labels": ["system"]},
    {"query": "lock the screen",                                "labels": ["system"]},
    {"query": "restart the computer",                           "labels": ["system"]},
    {"query": "turn on airplane mode",                          "labels": ["system"]},
    {"query": "connect to wifi",                                "labels": ["system"]},
    {"query": "empty the recycle bin",                          "labels": ["system"]},
    {"query": "turn off bluetooth",                             "labels": ["system"]},
    {"query": "check battery percentage",                       "labels": ["system"]},
    {"query": "what am i holding in my hand",                   "labels": ["vision"]},
    {"query": "what does this image show",                      "labels": ["vision"]},
    {"query": "read the text in this photo",                    "labels": ["vision"]},
    {"query": "what animal is this",                            "labels": ["vision"]},
    {"query": "what brand is this product",                     "labels": ["vision"]},
    {"query": "what is written on this whiteboard",             "labels": ["vision"]},
    {"query": "identify this plant for me",                     "labels": ["vision"]},
    {"query": "what car is this in the picture",                "labels": ["vision"]},
    {"query": "how many people are in this photo",              "labels": ["vision"]},
    {"query": "what is the color of this object",               "labels": ["vision"]},
]

test_data = test_data + extra_test_data

test_queries = [preprocess(item["query"]) for item in test_data]
test_labels = [item["labels"] for item in test_data]
X_test = vectorizer.transform(test_queries)
Y_test = mlb.transform(test_labels)

Y_pred = model.predict(X_test)

print(classification_report(Y_test, Y_pred, target_names=mlb.classes_, zero_division=0))
print(f"Overall Accuracy: {accuracy_score(Y_test, Y_pred):.2f}")