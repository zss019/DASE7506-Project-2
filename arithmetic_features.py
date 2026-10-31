"""Conservative numeric-equality checks. No symbolic algebra or code execution.
Unsupported syntax is skipped, never treated as an arithmetic error.
"""
import ast
from fractions import Fraction
import re
import numpy as np

DISPLAY = re.compile(r'(?<!\\)\$\$(.*?)\$\$|\\\[(.*?)\\\]|\\\((.*?)\\\)', re.S)
INLINE = re.compile(r'(?<![\\$])\$(?!\$)([^\n$]*?)(?<!\\)\$(?![\d$])')
FRAC = re.compile(r'\\(?:dfrac|tfrac|frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}')
WRAP = re.compile(r'\\boxed\s*\{([^{}]*)\}')
SURFACE_NAMES=['log_characters','log_words','log_numbers','log_equals','log_lines','log_checked_equations']
QUALITY_NAMES=['any_mismatch','mismatch_fraction','log_mismatches','log_rounded_equalities']


def normalize(expr):
    expr=expr.strip().replace(r'\$', '')
    for _ in range(10):
        new=FRAC.sub(r'((\1)/(\2))',expr)
        new=WRAP.sub(r'(\1)',new)
        if new==expr:break
        expr=new
    for token,replacement in [('\\times','*'),('\\cdot','*'),('\\div','/'),('×','*'),('÷','/'),('−','-'),('\\left',''),('\\right',''),('\\,',''),('\\;','')]:
        expr=expr.replace(token,replacement)
    expr=expr.replace('^','**').replace('{','(').replace('}',')')
    # Accept properly grouped thousands, not arbitrary comma deletion.
    expr=re.sub(r'(?<![\d,])\d{1,3}(?:,\d{3})+(?![\d,])',lambda m:m.group().replace(',',''),expr)
    return expr.strip()


def numeric(expr):
    expr=normalize(expr)
    if not expr or len(expr)>256 or not re.fullmatch(r'[\d.\s+*/()\-]+',expr):
        raise ValueError('Unsupported numeric expression')
    tree=ast.parse(expr,mode='eval')
    if sum(1 for _ in ast.walk(tree))>80:raise ValueError('Expression too large')
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            result=Fraction(str(node.value))
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            result=visit(node.operand)*(-1 if isinstance(node.op,ast.USub) else 1)
        elif isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add):result=a+b
            elif isinstance(node.op,ast.Sub):result=a-b
            elif isinstance(node.op,ast.Mult):result=a*b
            elif isinstance(node.op,ast.Div):result=a/b
            else:
                if b.denominator!=1 or abs(b)>6:raise ValueError('Exponent outside scope')
                result=a**int(b)
        else:raise ValueError('Unsupported AST')
        if abs(result)>10**15 or result.numerator.bit_length()>256 or result.denominator.bit_length()>256:
            raise ValueError('Value outside scope')
        return result
    return visit(tree.body),expr


def rounding_compatible(left_expr, right_expr, target):
    """Allow RHS-only displayed decimals to represent rounding intervals.
    Decimal constants repeated from the LHS remain exact. This is a conservative
    heuristic for intermediate rounding, not a proof of intended precision.
    """
    shared=set(re.findall(r'\d*\.\d+',left_expr))
    tree=ast.parse(right_expr,mode='eval')
    def interval(node):
        if isinstance(node,ast.Constant):
            token=ast.get_source_segment(right_expr,node)
            center=Fraction(str(node.value))
            width=Fraction(0)
            if '.' in token and token not in shared:
                width=Fraction(1,2*10**len(token.split('.')[1]))
            return center-width,center+width
        if isinstance(node,ast.UnaryOp):
            lo,hi=interval(node.operand)
            return (-hi,-lo) if isinstance(node.op,ast.USub) else (lo,hi)
        a,b=interval(node.left);c,d=interval(node.right)
        if isinstance(node.op,ast.Add):return a+c,b+d
        if isinstance(node.op,ast.Sub):return a-d,b-c
        if isinstance(node.op,ast.Mult):
            values=[a*c,a*d,b*c,b*d];return min(values),max(values)
        if isinstance(node.op,ast.Div):
            if c<=0<=d:raise ValueError('Uncertain zero denominator')
            values=[a/c,a/d,b/c,b/d];return min(values),max(values)
        if c!=d or c.denominator!=1 or abs(c)>6:raise ValueError('Uncertain exponent')
        if c<0 and a<=0<=b:raise ValueError('Uncertain zero base')
        values=[a**int(c),b**int(c)]
        if a<=0<=b and c>0 and int(c)%2==0:values.append(Fraction(0))
        return min(values),max(values)
    try:
        low,high=interval(tree.body)
        return low<=target<=high
    except (ValueError,ZeroDivisionError,OverflowError):
        return False


def equalities(text):
    chunks=[]
    def extract(source,regex):
        def collect(match):
            # Do not detach a math span from an operator outside its delimiter:
            # e.g. 4 x $3/4 = 3$ is NOT the equality 3/4 = 3.
            prefix=source[:match.start()].rsplit('\n',1)[-1]
            if not re.search(r'[+*/×÷−-]\s*$',prefix):
                chunks.append(next(g for g in match.groups() if g is not None))
            return '\n'
        return regex.sub(collect,source)
    remaining=extract(text,DISPLAY)
    remaining=extract(remaining,INLINE)
    # Plain Markdown bullets can be confused with unary minus: skip them.
    chunks.extend(line for line in remaining.splitlines() if not re.match(r'^\s*[-*+]\s+',line))
    records=[]
    for chunk in chunks:
        if '=' not in chunk:continue
        # A displayed environment may contain multiple independent lines.
        for line in chunk.splitlines():
            parts=line.strip().split('=')
            for left,right in zip(parts,parts[1:]):
                try:
                    a,la=numeric(left);b,rb=numeric(right)
                except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):
                    continue
                status='exact' if a==b else 'mismatch'
                if status=='mismatch' and rounding_compatible(la,rb,a):
                    status='rounded'
                records.append(dict(left=la,right=rb,status=status))
    return records


def text_features(text):
    records=equalities(text);checked=len(records)
    bad=sum(r['status']=='mismatch' for r in records);rounded=sum(r['status']=='rounded' for r in records)
    surface=np.log1p([len(text),len(text.split()),len(re.findall(r'\d+(?:\.\d+)?',text)),text.count('='),len(text.splitlines()),checked])
    quality=np.array([float(bad>0),bad/max(checked,1),np.log1p(bad),np.log1p(rounded)])
    return surface,quality,records


def bank_features(bank):
    surface=[];quality=[]
    for row in bank:
        sf=[];qf=[]
        for candidate in row['candidates']:
            a,b,_=text_features(candidate['text']);sf.append(a);qf.append(b)
        surface.append(sf);quality.append(qf)
    return np.asarray(surface),np.asarray(quality)
