"""Reference-derived arithmetic program supervision; train-only templates."""
import ast,re,json,hashlib
from fractions import Fraction
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from common import parse_answer
DIGITS=re.compile(r'(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])')
WORDS=dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety'.split(),list(range(21))+list(range(30,100,10))))
D=8192

def question_values(question):
 values=[]
 # Explicit standalone number words only; compound word-number phrases unsupported.
 for m in re.finditer(DIGITS.pattern+r'|\b(?:'+ '|'.join(WORDS)+r')\b',question,re.I):
  token=m.group().lower();v=Fraction(WORDS[token]) if token in WORDS else Fraction(token.replace(',',''))
  if v not in values:values.append(v)
 return values

def program(question,answer):
 values=question_values(question);env={};last=None
 if len(values)>8:return None
 for expr,result in re.findall(r'<<([^=<>]+)=([^<>]+)>>',answer):
  try:
   result=Fraction(result.replace(',',''));node=ast.parse(expr.replace(',',''),mode='eval').body
   def visit(n):
    if isinstance(n,ast.Constant) and type(n.value) in (int,float):
     v=Fraction(str(n.value))
     if v in values:return ['q',values.index(v)]
     if v in env:return env[v]
     if v in [1,2,100]:return ['c',str(v)]
     raise ValueError('unresolved literal')
    if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return ['-', ['c','0'],visit(n.operand)]
    if isinstance(n,ast.BinOp) and type(n.op) in [ast.Add,ast.Sub,ast.Mult,ast.Div]:
     op={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}[type(n.op)];a,b=visit(n.left),visit(n.right)
     if op in ['+','*'] and json.dumps(a)>json.dumps(b):a,b=b,a
     return [op,a,b]
    raise ValueError('unsupported AST')
   p=visit(node)
   if evaluate(p,values)!=result:continue
   if len(json.dumps(p))>240:continue
   env[result]=p;last=result
  except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,IndexError):continue
 gold=parse_answer(answer)
 return env.get(Fraction(gold)) if gold is not None else None

def evaluate(p,values):
 if p[0]=='q':return values[p[1]]
 if p[0]=='c':return Fraction(p[1])
 a,b=evaluate(p[1],values),evaluate(p[2],values)
 value={'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b,'/':lambda:a/b}[p[0]]()
 if abs(value)>10**12:raise ValueError('too large')
 return value

def text_matrix(questions):
 rows=[];cols=[];data=[]
 for i,q in enumerate(questions):
  tokens=re.findall(r'[a-z]+|\d+(?:\.\d+)?',q.lower());tokens=['#' if re.match(r'\d',t) else t for t in tokens]
  terms=tokens+[' '.join(tokens[k:k+2]) for k in range(len(tokens)-1)]+[' '.join(tokens[k:k+3]) for k in range(len(tokens)-2)]
  counts=Counter(int.from_bytes(hashlib.blake2b(t.encode(),digest_size=8).digest(),'little')%D for t in terms)
  norm=np.sqrt(sum((1+np.log(c))**2 for c in counts.values()))
  for col,c in counts.items():rows.append(i);cols.append(col);data.append((1+np.log(c))/max(norm,1))
 return csr_matrix((data,(rows,cols)),shape=(len(questions),D),dtype=np.float32)

def candidate_support(bank,templates,prob):
 out=np.zeros((len(bank),4));coverage=np.zeros(len(bank),bool)
 for i,q in enumerate(bank):
  values=question_values(q['question']);answers=[parse_answer(c['text']) for c in q['candidates']];mass={}
  for template,weight in zip(templates,prob[i]):
   try:v=str(evaluate(template,values));mass[v]=mass.get(v,0.)+float(weight)
   except (IndexError,ZeroDivisionError,ValueError,OverflowError):continue
  for j,a in enumerate(answers):out[i,j]=mass.get(a,0.) if a is not None else 0.
  coverage[i]=out[i].max()>0
 return out,coverage
