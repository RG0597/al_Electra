import torch
import flask
from flask import Flask,request,render_template
from flask_cors import CORS,cross_origin
import json
import albert.albert_xxlarge as albert
from rank import bm25_model
import util
import numpy as np
#from prediction import predict,init_model
from predElec import answer

app=Flask(__name__)
#model_electra=init_model()

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def get_answer():
    try:
        question=request.json['input_question']
        num_paragraphs=int(request.json['num_paragraphs'])
        question = question.lstrip().rstrip()
        link,text=util.get_url_text(question)
        if link!=None:
            bm_1, _, _ = bm25_model.get_similarity([question], text)
            bm_1 = np.array(bm_1)
            bm_1_idx = bm_1[bm_1[:, 1] > 1][:num_paragraphs, 0]
            bm_1_idx = np.array(bm_1_idx, dtype=int)
            text = ' '.join(text[i] for i in sorted(bm_1_idx))
            if len(bm_1_idx)==0:
                return app.response_class(response=json.dumps("Text passages not found. Provide more information in your question"),
                                          status=500,mimetype='application/json')

            res_albert=albert.answer(question,text)
            #_res_electra=predict(question,text,model_electra)
            elec=answer(question,text)
            #res_electra=_res_electra['q_0'][0]['text']if len(_res_electra)>0 else "answer not found"
            res = {'albert': res_albert,
                   'electra': elec,
                   'link': link,
                   'text_paragraphs': text}
            return flask.jsonify(res)
        else:
            return app.response_class(response=json.dumps("No wikipedia link found. Provide more information in your question"),
                                      status=500, mimetype='application/json')
    except Exception as e:
        res=str(e)
        return app.response_class(response=json.dumps(res),status=500,mimetype='application/json')


if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True,port=5000,use_reloader=False)