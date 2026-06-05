"""
build_math_tokens_v4.py — Math µMODEL data v4
  Format: Option 2 — simple Q:/A: pairs (no ### Instruction wrapper for math)
  New:    KXML numeric graphs for arithmetic (computation as graph nodes)
  Keep:   Filtered math JSONL records with ### Instruction/Response
  Output: tokens_math_v4.bin
"""
import json
import pathlib
import random
import re
import struct

import tiktoken

JSONL  = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\xshard_jsonl\prompt_math_layer.jsonl")
OUT    = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\tokens_math_v4.bin")
BLOCK  = 256
SEED   = 42
random.seed(SEED)
enc    = tiktoken.get_encoding("gpt2")

# ─── Math filter (same as v3) ─────────────────────────────────────────────────

MATH_TERMS = {
    'equation','solve','calculate','simplify','factor','expand','polynomial',
    'quadratic','linear','expression','variable','coefficient','exponent',
    'derivative','integral','differentiate','integrate','limit','converge',
    'triangle','circle','angle','radius','area','volume','hypotenuse',
    'pythagorean','probability','distribution','variance','mean','median',
    'prime','divisor','modulo','gcd','lcm','factorial','matrix','vector',
    'determinant','eigenvalue','function','domain','range','gradient',
    'proof','theorem','algebra','calculus','geometry','combinatorics',
    'permutation','combination','arithmetic','sequence','series',
}
MATH_SYM = re.compile(
    r'[=+\-×÷∫∂∑√πΣΠ∞≤≥≠≈]|\b\d+\s*[+\-×*/]\s*\d+|\b\d+\.\d+|f\([a-z]\)|x\s*=\s*[\d\-]')
GARBAGE  = ['base64','data:image','<img','iVBORw0KGgo','\x00','\xff\xd8','\x89PNG']

def is_math(text):
    tl = text.lower()
    return any(t in tl for t in MATH_TERMS) or bool(MATH_SYM.search(text)) or len(re.findall(r'\d+',text)) >= 3

def is_garbage(text):
    return any(s in text for s in GARBAGE) or bool(re.search(r'[A-Za-z0-9+/]{60,}={0,2}', text))

def normalize(p, r):
    p = re.sub(r'^[\s]*(Human|User|Q):\s*','', p, flags=re.IGNORECASE).strip()
    p = re.sub(r'^###\s*Instruction:\s*','', p, flags=re.IGNORECASE).strip()
    p = re.sub(r'\n+###\s*Response:\s*$','', p, flags=re.IGNORECASE).strip()
    r = re.sub(r'^[\s]*(Assistant|A):\s*','', r, flags=re.IGNORECASE).strip()
    p = p.replace('â€™',"'").replace('â€œ','"').replace('â€','"').replace('â€¦','...').replace('â€"','-').replace('â€"','-').replace('â€˜',"'")
    r = r.replace('â€™',"'").replace('â€œ','"').replace('â€','"').replace('â€¦','...').replace('â€"','-').replace('â€"','-').replace('â€˜',"'")
    p = p.replace('�','').strip()
    r = r.replace('�','').strip()
    return (p, r) if len(p)>=10 and len(r)>=15 else None

# ─── KXML Numeric Graph generators ───────────────────────────────────────────
# Each arithmetic operation is a computation graph:
#   <kxml:compute> → nodes (steps) → <result>
# Follows K'UHUL phase notation: Pop=init Sek=compute Ch'en=collect Xul=emit

def kxml_add(a, b):
    result = a + b
    sa, sb = str(a).zfill(max(len(str(a)),len(str(b)))), str(b).zfill(max(len(str(a)),len(str(b))))
    w = len(sa)
    carry, nodes = 0, []
    digits = []
    for i in range(w-1, -1, -1):
        d1, d2 = int(sa[i]), int(sb[i])
        s = d1+d2+carry
        carry = s//10
        digits.insert(0, str(s%10))
        col = w-i
        nodes.append(f'  <step phase="Sek" col="{col}">{d1}+{d2}+carry({carry if i<w-1 else 0})={s} write={s%10} carry={s//10}</step>')
    if carry:
        digits.insert(0, str(carry))
    res = ''.join(digits)
    graph = f'<kxml:compute op="add" a="{a}" b="{b}">\n'
    graph += f'  <step phase="Pop">align {a} and {b} right-justified</step>\n'
    graph += '\n'.join(nodes) + '\n'
    graph += f'  <result phase="Ch\'en">{res}</result>\n'
    graph += '</kxml:compute>'
    q = f"Q: What is {a} + {b}?\nA: {graph}\n{a} + {b} = {result}"
    return q

def kxml_sub(a, b):
    if b > a: a, b = b, a
    result = a - b
    sa = str(a).zfill(max(len(str(a)),len(str(b))))
    sb = str(b).zfill(max(len(str(a)),len(str(b))))
    w = len(sa)
    borrow, nodes, digits = 0, [], []
    for i in range(w-1, -1, -1):
        d1 = int(sa[i]) - borrow
        d2 = int(sb[i])
        if d1 < d2:
            d1 += 10; borrow = 1
        else:
            borrow = 0
        digit = d1 - d2
        digits.insert(0, str(digit))
        col = w-i
        nodes.append(f'  <step phase="Sek" col="{col}">{d1}-{d2}={digit}{"  borrow=1" if borrow else ""}</step>')
    res = ''.join(digits).lstrip('0') or '0'
    graph = f'<kxml:compute op="sub" a="{a}" b="{b}">\n'
    graph += f'  <step phase="Pop">align {a} and {b} right-justified</step>\n'
    graph += '\n'.join(nodes) + '\n'
    graph += f'  <result phase="Ch\'en">{res}</result>\n'
    graph += '</kxml:compute>'
    q = f"Q: What is {a} - {b}?\nA: {graph}\n{a} - {b} = {result}"
    return q

def kxml_mul(a, b):
    result = a * b
    sb = str(b)
    partial, nodes = [], []
    for i, d in enumerate(reversed(sb)):
        shift = i
        pp = int(d)*a*(10**shift)
        partial.append(pp)
        nodes.append(f'  <step phase="Sek" digit="{d}" shift="{shift}">{d}*{a}={int(d)*a} shifted={pp}</step>')
    pp_sum = '\n'.join(f'  <partial>{p}</partial>' for p in partial)
    graph = f'<kxml:compute op="mul" a="{a}" b="{b}">\n'
    graph += f'  <step phase="Pop">break {b} into digits</step>\n'
    graph += '\n'.join(nodes) + '\n'
    graph += pp_sum + '\n'
    graph += f'  <result phase="Ch\'en">{result}</result>\n'
    graph += '</kxml:compute>'
    q = f"Q: What is {a} * {b}?\nA: {graph}\n{a} * {b} = {result}"
    return q

def kxml_div(a, b):
    if b == 0: return None
    q_val = a // b
    rem   = a % b
    steps = []
    dividend = a
    quotient_digits = []
    partial = 0
    for d in str(a):
        partial = partial * 10 + int(d)
        qd = partial // b
        quotient_digits.append(str(qd))
        steps.append(f'  <step phase="Sek">bring down {d}: {partial} / {b} = {qd} remainder {partial - qd*b}</step>')
        partial = partial - qd * b
    quot_str = ''.join(quotient_digits).lstrip('0') or '0'
    graph = f'<kxml:compute op="div" a="{a}" b="{b}">\n'
    graph += f'  <step phase="Pop">long division {a} by {b}</step>\n'
    graph += '\n'.join(steps) + '\n'
    graph += f'  <result phase="Ch\'en">quotient={quot_str} remainder={rem}</result>\n'
    graph += '</kxml:compute>'
    if rem == 0:
        qa = f"Q: What is {a} / {b}?\nA: {graph}\n{a} / {b} = {quot_str}"
    else:
        qa = f"Q: What is {a} / {b}?\nA: {graph}\n{a} / {b} = {quot_str} remainder {rem}"
    return qa

def generate_kxml_graphs(n=2500):
    records = []
    for _ in range(n):
        records.append(kxml_add(random.randint(1,9999), random.randint(1,9999)))
    for _ in range(n):
        a = random.randint(10,9999); b = random.randint(1,a)
        records.append(kxml_sub(a, b))
    for _ in range(n):
        records.append(kxml_mul(random.randint(2,999), random.randint(2,99)))
    for _ in range(n//2):
        a = random.randint(2,999); b = random.randint(2,50)
        r = kxml_div(a*b + random.randint(0,b-1), b)   # some with remainder
        if r: records.append(r)
        r2 = kxml_div(a*b, b)  # clean division
        if r2: records.append(r2)
    random.shuffle(records)
    return records

# ─── Simple Q/A facts (Option 2) ─────────────────────────────────────────────
# Short direct pairs the model can pattern-match at inference

def generate_direct_qa(n=1000):
    records = []
    for _ in range(n):
        a, b = random.randint(1,999), random.randint(1,999)
        records.append(f"Q: {a} + {b} =\nA: {a+b}")
        records.append(f"Q: {a} - {b} =\nA: {a-b}")
        a2, b2 = random.randint(2,99), random.randint(2,12)
        records.append(f"Q: {a2} * {b2} =\nA: {a2*b2}")
        if b2 != 0 and a2*b2 <= 9999:
            records.append(f"Q: {a2*b2} / {b2} =\nA: {a2}")
        records.append(f"Q: What is {a} plus {b}?\nA: {a+b}")
        records.append(f"Q: What is {a} minus {b}?\nA: {a-b}")
        records.append(f"Q: What is {a2} times {b2}?\nA: {a2*b2}")
    random.shuffle(records)
    return records

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building math data v4 (KXML numeric graphs + Q/A format)...")

    # 1. Filter JSONL
    print("  Loading + filtering JSONL...")
    jsonl_recs, stats = [], {'total':0,'garbage':0,'non_math':0,'format_err':0,'good':0}
    with open(JSONL, encoding='utf-8', errors='replace') as f:
        for line in f:
            stats['total'] += 1
            try:
                r = json.loads(line)
                p, resp = r.get('prompt',''), r.get('response','')
                if not p or not resp: continue
                if is_garbage(p) or is_garbage(resp): stats['garbage']+=1; continue
                if not is_math(p+' '+resp): stats['non_math']+=1; continue
                result = normalize(p, resp)
                if not result: stats['format_err']+=1; continue
                inst, ans = result
                jsonl_recs.append(f"### Instruction:\n{inst}\n\n### Response:\n{ans}<|endoftext|>")
                stats['good'] += 1
            except: stats['format_err']+=1

    print(f"  JSONL: {stats['good']} good / {stats['garbage']} binary / {stats['non_math']} non-math")

    # 2. KXML numeric graphs
    print("  Generating KXML numeric graphs...")
    kxml_recs = generate_kxml_graphs(n=2500)
    print(f"  KXML: {len(kxml_recs)} graphs (add/sub/mul/div)")

    # 3. Direct Q/A (Option 2)
    print("  Generating direct Q/A pairs...")
    qa_recs = generate_direct_qa(n=1000)
    print(f"  Direct Q/A: {len(qa_recs)} pairs")

    # 4. Combine + tokenize
    all_recs = jsonl_recs + kxml_recs + qa_recs
    random.shuffle(all_recs)
    print(f"  Total: {len(all_recs)} records")

    print("  Tokenizing...")
    toks = []
    for text in all_recs:
        toks.extend(enc.encode(text, allowed_special={"<|endoftext|>"}))

    seqs = len(toks) // BLOCK
    flat = toks[:seqs*BLOCK]
    with open(OUT, 'wb') as f:
        f.write(struct.pack('<II', seqs, BLOCK))
        f.write(struct.pack(f'<{len(flat)}I', *flat))

    print()
    print("=" * 60)
    print("  tokens_math_v4.bin")
    print(f"  {seqs:,} seqs x {BLOCK}  ({OUT.stat().st_size/1e6:.1f} MB)")
    print(f"  1 epoch at batch=4: {seqs//4:,} steps")
    print(f"  KXML graphs: {len(kxml_recs)}  Direct Q/A: {len(qa_recs)}")
    print("=" * 60)

if __name__ == '__main__':
    main()
