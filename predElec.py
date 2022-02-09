import torch
from transformers import AutoModelForQuestionAnswering,AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('deepset/electra-base-squad2')
model=AutoModelForQuestionAnswering.from_pretrained('deepset/electra-base-squad2')

def answer(question,text):
    input_dict=tokenizer.encode_plus(question,text,return_tensors='pt',max_length=512)
    input_ids=input_dict["input_ids"].tolist()
    outputs=model(**input_dict)
    start=torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits)

    all_tokens=tokenizer.convert_ids_to_tokens(input_ids[0])
    answer=''.join(all_tokens[start:end+1]).replace('_','').strip()
    answer=answer.replace('[SEP]','')
    return answer if answer !='[CLS]' and len(answer)!=0 else 'answer not found'

