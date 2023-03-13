# nf_classifier
News Frames Classifier

```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-0.txt
pip install -r requirements-1.txt
pip install -r requirements-2.txt

src/ner_train.py -c 0 -l 2e-5 -e 40 -b 20 sl hr sr bs mk sq cs sk pl bg uk ru mcbert
src/ner_train.py -c 0 -l 2e-5 -e 40 -b 24 sl hr sr bs mk sq cs sk pl bg uk ru xlmrb
src/ner_train.py -c 0 -l 2e-5 -e 40 -b 24 sl hr sr bs mk sq cs sk pl bg uk ru xlmrl
```