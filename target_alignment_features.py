"""Deterministic lexical proxies for requested quantity alignment; no external models."""
import re
import numpy as np
STOP=set('a an the is are was were be been being do does did how many much what which who when where why of to in on at for and or if then than by with from as it its his her their he she they them this that these those will would can could should have has had there total number amount'.split())
CUES={
 'difference':r'\b(more|less|fewer|difference|exceed)\b',
 'remaining':r'\b(left|remain\w*|rest|after)\b',
 'aggregate':r'\b(total|altogether|combined|together|all)\b',
 'rate':r'\b(per|each|every|average)\b',
 'percentage':r'\b(percent|percentage)\b|%',
 'money':r'\b(dollar\w*|cent\w*|cost\w*|price\w*|money|earn\w*|spend\w*|spent|profit)\b|\$',
 'time':r'\b(second\w*|minute\w*|hour\w*|day\w*|week\w*|month\w*|year\w*)\b',
 'distance':r'\b(mile\w*|kilometer\w*|meter\w*|feet|foot|inch\w*)\b'}
GENERIC=['log_solution_words','log_tail_words','question_mark_in_solution','explicit_final_marker']
TARGET=['target_tail_recall','target_tail_precision','target_body_recall','tail_minus_body_recall','target_bigram_tail_recall','target_name_tail_recall']+[f'{k}_{v}' for k in CUES for v in ['matched','missing']]

def tokens(s):
    words=re.findall(r'[a-z]+',s.lower())
    return [w[:-1] if len(w)>3 and w.endswith('s') else w for w in words if w not in STOP]

def target_clause(q):
    starts=list(re.finditer(r'\b(?:how many|how much|how long|how far|what|which|who)\b',q,re.I))
    return q[starts[-1].start():] if starts else re.split(r'(?<=[.!?])\s+',q)[-1]

def features(bank):
    result=np.zeros((len(bank),4,len(GENERIC)+len(TARGET)))
    for i,q in enumerate(bank):
        target=target_clause(q['question']);tw=tokens(target);ts=set(tw)
        big=set(zip(tw,tw[1:]));names=set(re.findall(r'\b[A-Z][a-z]+\b',target))-{'How','What','Which','Who','If'}
        for j,c in enumerate(q['candidates']):
            text=c['text'];body=text.split('####')[0].strip()
            sentences=re.split(r'(?<=[.!?])\s+|\n+',body)
            tail=' '.join(sentences[-2:]);tt=tokens(tail);ss=set(tt);bs=set(tokens(body))
            recall=len(ts&ss)/max(len(ts),1);br=len(ts&bs)/max(len(ts),1)
            row=[np.log1p(len(text.split())),np.log1p(len(tail.split())),float('?' in text),float('####' in text),recall,len(ts&ss)/max(len(ss),1),br,recall-br,len(big&set(zip(tt,tt[1:])))/max(len(big),1),sum(bool(re.search(r'\b'+re.escape(w)+r'\b',tail)) for w in names)/max(len(names),1)]
            for pattern in CUES.values():
                asked=bool(re.search(pattern,target,re.I));mentioned=bool(re.search(pattern,tail,re.I))
                row.extend([float(asked and mentioned),float(asked and not mentioned)])
            result[i,j]=row
    return result
