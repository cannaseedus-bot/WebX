"""
build_kxml_unified.py — Universal KXML dataset builder
Applies KXML graph format to ALL three data layers:
  math    → <kxml:compute op="calculate"> with full column arithmetic steps
  code    → <kxml:compute op="generate" domain="code">
  instruct→ <kxml:compute op="reason" domain="general">

Why KXML for everything:
  Every record becomes a K'UHUL phase program.
  The model simultaneously learns:
    1. Domain knowledge (math, code, general reasoning)
    2. K'UHUL phase system: Pop→Wo→Sek→Ch'en→Xul
    3. KXML graph notation (compute nodes, edges, phases)
  This is chain-of-thought through structure — not added tokens,
  but the format itself encodes the reasoning process.

K'UHUL phase → arithmetic/reasoning mapping:
  Pop    understand problem, gather inputs
  Wo     declare intent, choose method
  Sek    execute computation (the actual work)
  Ch'en  collect result, verify
  Xul    emit final answer

Output: tokens_kxml_unified.bin
"""
import json, re, struct, pathlib, random, textwrap
import tiktoken

MATH_JSONL    = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\xshard_jsonl\prompt_math_layer.jsonl")
INSTRUCT_JSONL= pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\xshard_jsonl\prompt_instruct_layer.jsonl")
CODE_JSONL    = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\xshard_jsonl\prompt_code_layer.jsonl")
OUT           = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\tokens_kxml_unified.bin")
BLOCK         = 256
SEED          = 42
random.seed(SEED)
enc = tiktoken.get_encoding("gpt2")

# ─── Filters (same as v4) ─────────────────────────────────────────────────────

MATH_TERMS = {
    'equation','solve','calculate','simplify','factor','expand','polynomial',
    'quadratic','linear','expression','variable','coefficient','exponent',
    'derivative','integral','differentiate','integrate','limit','converge',
    'triangle','circle','angle','radius','area','volume','probability',
    'distribution','variance','mean','median','prime','matrix','vector',
    'determinant','function','domain','range','gradient','proof','theorem',
    'algebra','calculus','geometry','combinatorics','permutation','arithmetic',
}
MATH_SYM = re.compile(
    r'[=+\-×÷∫∂∑√πΣΠ∞≤≥≠≈]|\b\d+\s*[+\-×*/]\s*\d+|\b\d+\.\d+|f\([a-z]\)')
GARBAGE = ['base64','data:image','<img','iVBORw0KGgo','\x00','\xff\xd8','\x89PNG']

def is_garbage(text):
    return any(s in text for s in GARBAGE) or bool(re.search(r'[A-Za-z0-9+/]{60,}={0,2}',text))

def is_math(text):
    tl = text.lower()
    return any(t in tl for t in MATH_TERMS) or bool(MATH_SYM.search(text)) or len(re.findall(r'\d+',text))>=3

def normalize(p, r):
    for pat in [r'^[\s]*(Human|User|Q):\s*', r'^###\s*Instruction:\s*', r'\n+###\s*Response:\s*$']:
        p = re.sub(pat,'',p,flags=re.IGNORECASE).strip()
    r = re.sub(r'^[\s]*(Assistant|A):\s*','',r,flags=re.IGNORECASE).strip()
    for fix in [('â€™',"'"),('â€œ','"'),('â€','"'),('â€¦','...'),('â€"','-'),('â€˜',"'"),('â\x80\x99',"'"),('â\x80\x9c','"'),('â\x80\x9d','"')]:
        p=p.replace(*fix); r=r.replace(*fix)
    p=p.replace('â','').replace('€','').replace('™','').replace('œ','').replace('¦','').replace('˜','')
    r=r.replace('â','').replace('€','').replace('™','').replace('œ','').replace('¦','').replace('˜','')
    p=p.replace('�','').strip(); r=r.replace('�','').strip()
    return (p,r) if len(p)>=10 and len(r)>=15 else None

# ─── KXML wrappers ────────────────────────────────────────────────────────────

def kxml_math_exact(a_op, op_name, result, steps_xml):
    """Full computation graph for arithmetic with explicit steps."""
    return f"""<kxml:compute op="{op_name}" domain="math" phase="Sek">
  <step phase="Pop">input: {a_op}</step>
{steps_xml}  <result phase="Ch'en">{result}</result>
</kxml:compute>"""

def kxml_general(question, answer, domain="general"):
    """KXML wrapper for any Q/A pair — teaches K'UHUL phase structure."""
    # Auto-detect a Wo (plan) from the question
    q_lower = question.lower()
    if any(w in q_lower for w in ['prove','show','derive']):
        method = "proof/derivation"
    elif any(w in q_lower for w in ['solve','find','calculate','compute']):
        method = "direct computation"
    elif any(w in q_lower for w in ['explain','describe','what is']):
        method = "explanation"
    elif any(w in q_lower for w in ['write','generate','create','implement']):
        method = "generation"
    else:
        method = "analysis"
    # Truncate long answers for Sek step summary
    sek = answer[:120].replace('\n',' ').strip()
    if len(answer) > 120: sek += '...'
    return f"""Q: {question}
A: <kxml:compute op="respond" domain="{domain}" phase="Sek">
  <step phase="Pop">understand: {question[:80].strip()}</step>
  <step phase="Wo">method: {method}</step>
  <step phase="Sek">{sek}</step>
  <result phase="Ch'en">{answer}</result>
</kxml:compute>"""

# ─── Arithmetic KXML generators (same logic as v4, KXML format) ───────────────

def add_kxml(a, b):
    r = a+b
    sa,sb = str(a).zfill(max(len(str(a)),len(str(b)))), str(b).zfill(max(len(str(a)),len(str(b))))
    w=len(sa); carry=0; nodes=[]; digits=[]
    for i in range(w-1,-1,-1):
        d1,d2=int(sa[i]),int(sb[i]); s=d1+d2+carry; carry=s//10
        digits.insert(0,str(s%10))
        nodes.append(f'  <step phase="Sek" col="{w-i}">{d1}+{d2}+{carry if i<w-1 else 0}={s} write={s%10} carry={s//10}</step>')
    if carry: digits.insert(0,str(carry))
    steps=f'  <step phase="Pop">align right-justified: {sa} + {sb}</step>\n'+'\n'.join(nodes)+'\n'
    g=kxml_math_exact(f"{a}+{b}","add",r,steps)
    return f"Q: What is {a} + {b}?\nA: {g}\n{a} + {b} = {r}"

def sub_kxml(a, b):
    if b>a: a,b=b,a
    r=a-b; sa=str(a).zfill(max(len(str(a)),len(str(b)))); sb=str(b).zfill(max(len(str(a)),len(str(b))))
    w=len(sa); borrow=0; nodes=[]; digits=[]
    for i in range(w-1,-1,-1):
        d1=int(sa[i])-borrow; d2=int(sb[i])
        if d1<d2: d1+=10; borrow=1
        else: borrow=0
        digits.insert(0,str(d1-d2))
        nodes.append(f'  <step phase="Sek" col="{w-i}">{d1}-{d2}={d1-d2}{"  borrow=1" if borrow else ""}</step>')
    res=''.join(digits).lstrip('0') or '0'
    steps=f'  <step phase="Pop">align right-justified: {sa} - {sb}</step>\n'+'\n'.join(nodes)+'\n'
    g=kxml_math_exact(f"{a}-{b}","sub",res,steps)
    return f"Q: What is {a} - {b}?\nA: {g}\n{a} - {b} = {r}"

def mul_kxml(a, b):
    r=a*b; sb=str(b); partial=[]; nodes=[]
    for i,d in enumerate(reversed(sb)):
        pp=int(d)*a*(10**i); partial.append(pp)
        nodes.append(f'  <step phase="Sek" digit="{d}" shift="{i}">{d}*{a}={int(d)*a} shifted={pp}</step>')
    pp_xml=''.join(f'  <partial>{p}</partial>\n' for p in partial)
    steps=f'  <step phase="Pop">break {b} into digits for long multiplication</step>\n'+'\n'.join(nodes)+'\n'+pp_xml
    g=kxml_math_exact(f"{a}*{b}","mul",r,steps)
    return f"Q: What is {a} * {b}?\nA: {g}\n{a} * {b} = {r}"

def div_kxml(a, b):
    if b==0: return None
    q_val,rem=a//b,a%b; steps=[]; partial=0; qd_list=[]
    for d in str(a):
        partial=partial*10+int(d); qd=partial//b
        steps.append(f'  <step phase="Sek">bring down {d}: {partial}/{b}={qd} r={partial-qd*b}</step>')
        qd_list.append(str(qd)); partial=partial-qd*b
    quot=''.join(qd_list).lstrip('0') or '0'
    res=f"{quot} r{rem}" if rem else quot
    step_xml=f'  <step phase="Pop">long division {a} / {b}</step>\n'+'\n'.join(steps)+'\n'
    g=kxml_math_exact(f"{a}/{b}","div",res,step_xml)
    suffix=f"{quot} remainder {rem}" if rem else quot
    return f"Q: What is {a} / {b}?\nA: {g}\n{a} / {b} = {suffix}"

# ─── Direct Q/A (Option 2 — short format for inference) ──────────────────────

def direct_qa(n=800):
    recs=[]
    for _ in range(n):
        a,b=random.randint(1,9999),random.randint(1,9999)
        recs.append(f"Q: {a} + {b} =\nA: {a+b}")
        recs.append(f"Q: {a} - {b} =\nA: {a-b}")
        a2,b2=random.randint(2,99),random.randint(2,12)
        recs.append(f"Q: {a2} * {b2} =\nA: {a2*b2}")
        if b2: recs.append(f"Q: {a2*b2} / {b2} =\nA: {a2}")
        recs.append(f"Q: What is {a} plus {b}?\nA: {a+b}")
        recs.append(f"Q: What is {a2} times {b2}?\nA: {a2*b2}")
    random.shuffle(recs); return recs

# ─── Load + wrap one JSONL file ───────────────────────────────────────────────

def load_jsonl(path, domain, max_recs=None, math_filter=False):
    recs=[]; skipped=0
    with open(path, encoding='utf-8', errors='replace') as f:
        for i,line in enumerate(f):
            if max_recs and i>=max_recs: break
            try:
                r=json.loads(line)
                p,resp=r.get('prompt',''),r.get('response','')
                if not p or not resp: continue
                if is_garbage(p) or is_garbage(resp): skipped+=1; continue
                if math_filter and not is_math(p+' '+resp): skipped+=1; continue
                result=normalize(p,resp)
                if not result: skipped+=1; continue
                inst,ans=result
                recs.append(kxml_general(inst,ans,domain=domain))
            except: skipped+=1
    print(f"    {path.name}: {len(recs)} good, {skipped} skipped")
    return recs

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building KXML unified dataset...")
    all_records=[]

    # 1. Math JSONL → KXML general wrapper
    print("  [math]")
    all_records += load_jsonl(MATH_JSONL, domain="math", math_filter=True)

    # 2. Instruct JSONL → KXML general wrapper
    print("  [instruct]")
    all_records += load_jsonl(INSTRUCT_JSONL, domain="general")

    # 3. Code JSONL → KXML general wrapper
    print("  [code]")
    all_records += load_jsonl(CODE_JSONL, domain="code")

    # 4. Arithmetic KXML graphs (full step-by-step)
    print("  [arithmetic KXML graphs]")
    arith=[]
    for _ in range(2500): arith.append(add_kxml(random.randint(1,9999),random.randint(1,9999)))
    for _ in range(2500):
        a=random.randint(10,9999); b=random.randint(1,a); arith.append(sub_kxml(a,b))
    for _ in range(2500): arith.append(mul_kxml(random.randint(2,999),random.randint(2,99)))
    for _ in range(1250):
        a,b=random.randint(2,999),random.randint(2,50)
        r=div_kxml(a*b+random.randint(0,b-1),b)
        if r: arith.append(r)
        r2=div_kxml(a*b,b)
        if r2: arith.append(r2)
    print(f"    {len(arith)} arithmetic graphs")
    all_records += arith

    # 5. Direct Q/A (Option 2 — teaches model to answer at inference time)
    print("  [direct Q/A pairs]")
    qa=direct_qa(800)
    print(f"    {len(qa)} pairs")
    all_records += qa

    random.shuffle(all_records)
    print(f"\n  Total records: {len(all_records)}")

    # Tokenize
    print("  Tokenizing...")
    toks=[]
    for text in all_records:
        toks.extend(enc.encode(text, allowed_special={"<|endoftext|>"}))

    seqs=len(toks)//BLOCK
    flat=toks[:seqs*BLOCK]
    with open(OUT,'wb') as f:
        f.write(struct.pack('<II',seqs,BLOCK))
        f.write(struct.pack(f'<{len(flat)}I',*flat))

    print()
    print("="*60)
    print(f"  tokens_kxml_unified.bin")
    print(f"  {seqs:,} seqs x {BLOCK}  ({OUT.stat().st_size/1e6:.1f} MB)")
    print(f"  1 epoch at batch=4: {seqs//4:,} steps")
    print()
    print("  Layers:")
    print(f"    math JSONL (KXML wrapped)    domain=math")
    print(f"    instruct JSONL (KXML wrapped) domain=general")
    print(f"    code JSONL (KXML wrapped)     domain=code")
    print(f"    arithmetic KXML graphs        {len(arith)} records")
    print(f"    direct Q/A pairs              {len(qa)} records")
    print()
    print("  Model trains on K'UHUL simultaneously:")
    print("    Pop  = understand input")
    print("    Wo   = declare method")
    print("    Sek  = execute computation")
    print("    Ch'en= collect result")
    print("    Xul  = emit answer")
    print("="*60)

if __name__=='__main__':
    main()
