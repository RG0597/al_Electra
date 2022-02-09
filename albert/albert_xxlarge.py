import torch
from transformers import AlbertTokenizer,AlbertForQuestionAnswering
tokenizer=AlbertTokenizer.from_pretrained('ahotrod/albert_xxlargev1_squad2_512')
model=AlbertForQuestionAnswering.from_pretrained('ahotrod/albert_xxlargev1_squad2_512')

def answer(question,text):
    input_dict=tokenizer.encode_plus(question,text,return_tensors='pt',max_length=512)
    input_ids=input_dict["input_ids"].tolist()
    outputs=model(**input_dict)
    start=torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits)

    all_tokens=tokenizer.convert_ids_to_tokens(input_ids[0])
    answer=''.join(all_tokens[start:end+1]).strip()
    answer=answer.replace('[SEP]','')
    print(answer)
    answer=answer.replace('_','')
    answer = answer.replace('__', '')
    return answer[1:] if answer !='[CLS]' and len(answer)!=0 else 'answer not found'

