import os
import json
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from ibm_watson import SpeechToTextV1, NaturalLanguageUnderstandingV1, ApiException
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
from functools import reduce 

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'flac'}
MODEL_ID=''
USE_MOCK = False

RECOMMENDATIONS = [
    {
     'CARRO': 'TORO',
     'SEGURANCA': 3,
     'CONSUMO': 6,
     'DESEMPENHO': 4,
     'MANUTENCAO': 1,
     'CONFORTO': 4,
     'DESIGN': 6,
     'ACESSORIOS': 4,
     'MODELO': 0
    },

    {
     'CARRO': 'DUCATO',
     'SEGURANCA': 1,
     'CONSUMO': 5,
     'DESEMPENHO': 6,
     'MANUTENCAO': 0,
     'CONFORTO': 5,
     'DESIGN': 3,
     'ACESSORIOS': 2,
     'MODELO': 0
    },

    {
     'CARRO': 'FIORINO',
     'SEGURANCA': 0,
     'CONSUMO': 4,
     'DESEMPENHO': 0,
     'MANUTENCAO': 3,
     'CONFORTO': 1,
     'DESIGN': 1,
     'ACESSORIOS': 0,
     'MODELO': 0
    },

    {
     'CARRO': 'CRONOS',
     'SEGURANCA': 2,
     'CONSUMO': 3,
     'DESEMPENHO': 7,
     'MANUTENCAO': 0,
     'CONFORTO': 7,
     'DESIGN': 9,
     'ACESSORIOS': 4,
     'MODELO': 0
    },

    {
     'CARRO': 'FIAT 500',
     'SEGURANCA': 5,
     'CONSUMO': 3,
     'DESEMPENHO': 1,
     'MANUTENCAO': 2,
     'CONFORTO': 4,
     'DESIGN': 10,
     'ACESSORIOS': 4,
     'MODELO': 0
    },

    {
     'CARRO': 'MAREA',
     'SEGURANCA': 3,
     'CONSUMO': 3,
     'DESEMPENHO': 7,
     'MANUTENCAO': 0,
     'CONFORTO': 6,
     'DESIGN': 3,
     'ACESSORIOS': 1,
     'MODELO': 0
    },

    {
     'CARRO': 'LINEA',
     'SEGURANCA': 1,
     'CONSUMO': 1,
     'DESEMPENHO': 4,
     'MANUTENCAO': 0,
     'CONFORTO': 3,
     'DESIGN': 2,
     'ACESSORIOS': 1,
     'MODELO': 0
    },

    {
     'CARRO': 'ARGO',
     'SEGURANCA': 2,
     'CONSUMO': 9,
     'DESEMPENHO': 6,
     'MANUTENCAO': 2,
     'CONFORTO': 7,
     'DESIGN': 8,
     'ACESSORIOS': 9,
     'MODELO': 0
    },

    {
     'CARRO': 'RENEGADE',
     'SEGURANCA': 0,
     'CONSUMO': 1,
     'DESEMPENHO': 0,
     'MANUTENCAO': 0,
     'CONFORTO': 2,
     'DESIGN': 2,
     'ACESSORIOS': 2,
     'MODELO': 0
    }
]

ENTITIES = {
    'SEGURANCA': 0.0,
    'CONSUMO': 0.0,
    'DESEMPENHO': 0.0,
    'MANUTENCAO': 0.0,
    'CONFORTO': 0.0,
    'DESIGN': 0.0,
    'ACESSORIOS': 0.0,
    'MODELO': 0.0
}
   
@app.route("/")
def index():
  return jsonify(api='/api')

@app.route("/api", methods=['POST'])
def api():
    car = request.form.get('car')  
    text = request.form.get('text')
    if car and 'audio' in request.files:
        file = request.files['audio']
        if file.filename and allowed_file(file.filename):
            return run_audio(car, file)
        else:
            return default_response()
    elif car and text:
        return run_text(car, text)   
    else:
        return default_response()
 
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_audio(car, file):
    save_file(file)
    response = speech_to_text(file)
    remove_file(file)
    if response:
        text = prepare_speech(response)
        return execute_rules(car, text)
    else:
        return default_response()

def run_text(car, text):
    return execute_rules(car, text)

def save_file(file):
    filename = secure_filename(file.filename)
    file.save(os.path.join(UPLOAD_FOLDER, filename))

def remove_file(file):
    filename = secure_filename(file.filename)
    os.remove(os.path.join(UPLOAD_FOLDER, filename))

def speech_to_text(file):
    try:
        if USE_MOCK:
            return speech_mock()
        else:
            with open(os.path.join(UPLOAD_FOLDER, file.filename), 'rb') as audio_file:
                service = SpeechToTextV1()
                return service.recognize(
                        audio=audio_file,
                        content_type='audio/flac',
                        model='pt-BR_NarrowbandModel',
                        max_alternatives=0
                ).get_result()
                
    except ApiException as e:
       print(e.global_transaction_id)
       return None

def prepare_speech(response):
    results = []
    for r in response['results']:
        for a in r['alternatives']:
            results.append(a['transcript'])
    return ' '.join(results)

def prepare_nlu(response):
    results = []
    for e in response['entities']:
        if e['confidence'] > 0.60:
            results.append({
                'entity': e['type'],
                'sentiment': e['sentiment']['score'],
                'mention': e['text']
            })        
    return results
    
def run_nlu(text):
    try:
        if USE_MOCK:
            return nlu_mock()
        else:
            service = NaturalLanguageUnderstandingV1(version='2020-09-13')
            return service.analyze(
                text=text,
                language='pt',
                features=Features(
                    entities=EntitiesOptions(model=MODEL_ID, mentions=False, sentiment=True),
                    keywords=None
                ),
                version='2020-09-13'
            ).get_result()
        
    except ApiException as e:
        print(e.global_transaction_id)
        return None

def default_response():
    return jsonify(recommendation='', entities=[])

def execute_rules(car, text):
    nlu = run_nlu(text)
    if nlu: 
        result = prepare_nlu(nlu)
        if sentiment_score(result) >= 0:
            return default_response()
        else:           
            return recommendation(car, result)
    else:
        return default_response()

def sentiment_score(result):
    return reduce(lambda acc, val: acc + val['sentiment'], result, 0)

def accumulate_entities(acc, val):
    acc[val['entity']] += val['sentiment']
    return acc

def recommendation(car, result):
    entities = reduce(accumulate_entities, result, ENTITIES)
    target_key = min(entities, key=entities.get)
    target_value = entities[target_key]
    similarities = get_similarities(entities, target_key, target_value)
    absolute_difference = get_absolute_difference(entities)
    if similarities or absolute_difference < 0.1:
        # Recommend according to the priority table
        if similarities:
            items = [key for (key, value) in similarities.items()]
            items.insert(0, target_key)
            return get_recommendation_by(car, items, result)
        else:
            items = [key for (key, value) in entities.items() if value < 0.0]         
            return get_recommendation_by(car, items, result)
    else:
        # Recommend most negative one
        return get_recommendation_by(car, [target_key], result)
    
def get_similarities(entities, key, value):
    items = filter(lambda i: i[0] != key and i[1] == value, entities.items())
    return {key: value for (key, value) in items}

def get_absolute_difference(entities):
    items = [value for (key, value) in entities.items() if value < 0.0]
    return abs(reduce(lambda acc, val: acc - val, items))

def get_recommendation_by(car, items, result):  
    recommendations = filter(lambda r: r['CARRO'].lower() != car.lower(), RECOMMENDATIONS)
    cars = sorted(recommendations, key = lambda i: [i[item] for item in items], reverse=True)
    return jsonify({
        'recommendation': cars[0]['CARRO'],
        'entities': result
    })

def speech_mock():
    return {
        'result_index': 0, 
        'results': [
            {
                'final': True, 
                'alternatives': [
                    {
                        'transcript': 'espetacular visual é lindíssimo com belas amente kit multimídia compatível com android auto ', 
                        'confidence': 0.84
                    }
                ]
            }, 
            {
                'final': True, 
                'alternatives': [
                    {
                        'transcript': 'o consumo destes já passa de sete novos quilômetros isso na cidade ', 
                        'confidence': 0.55
                    }
                ]
            }
        ]
    }

def nlu_mock():
    return {
    "entities": [
        {
        "confidence": 0.643092,
        "count": 1,
        "disambiguation": {
            "subtype": [
            "NONE"
            ]
        },
        "sentiment": {
            "label": "negative",
            "score": -0.851488
        },
        "text": "mais feio",
        "type": "DESIGN"
        },
        {
        "confidence": 0.556987,
        "count": 1,
        "disambiguation": {
            "subtype": [
            "NONE"
            ]
        },
        "sentiment": {
            "label": "negative",
            "score": -0.851488
        },
        "text": "minha vida",
        "type": "DESIGN"
        }
    ],
    "language": "pt",
    "usage": {
        "features": 1,
        "text_characters": 68,
        "text_units": 1
    }
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)