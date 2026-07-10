#!/usr/bin/env python3
"""Needle-in-a-haystack retest with a NEUTRAL needle (no 'secret/passcode' framing that
triggered the model to refuse to echo it). Focus on the lengths that were confounded."""
import json, time, urllib.request, sys

URL="http://localhost:8000/v1/chat/completions"; TOK="http://localhost:8000/tokenize"; MODEL="MiMo-V2.5"
FILLER=[
 "The harbor town woke slowly under a pale sky, and the fishing boats creaked against the pier.",
 "Economists debated the quarterly figures while the interns refilled the coffee for the third time.",
 "In the highlands, the shepherds moved their flocks before the first frost settled on the passes.",
 "The library's east wing held maps no cartographer had updated in nearly ninety years.",
 "Rain traced long lines down the window as the train pushed north through the empty fields.",
 "She catalogued each mineral by hardness, streak, and the faint smell of the fractured rock.",
 "The committee postponed the vote, citing the unfinished audit and a missing signature.",
 "Far offshore, the buoys blinked in sequence, indifferent to the storm gathering behind them.",
]
def post(u,o,t=1200):
    r=urllib.request.Request(u,data=json.dumps(o).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=t) as f: return json.load(f)
def ntok(t):
    try: return post(TOK,{"model":MODEL,"prompt":t},120).get("count")
    except: return None
samp=" ".join(FILLER*40); tpc=(ntok(samp) or len(samp)/4)/len(samp)
print(f"# tokens/char={tpc:.4f}")

# NEUTRAL needle: a benign counted quantity, nothing 'secret'.
NUM="738415962"
NEEDLE=f" At the Lisbon spring festival, the volunteers counted exactly {NUM} paper lanterns floating on the river. "
Q=("According to the document, exactly how many paper lanterns did the volunteers count at the "
   "Lisbon spring festival? Reply with only the number.")

def build(target,depth):
    body=max(target-200,100); approx=int(body/tpc); buf=[];n=0;i=0
    while n<approx:
        s=FILLER[i%len(FILLER)]; buf.append(s); n+=len(s)+1; i+=1
    t=" ".join(buf); cut=int(len(t)*depth); sp=t.find(" ",cut); sp=sp if sp!=-1 else cut
    return t[:sp]+NEEDLE+t[sp:]

PLAN=[(8_000,[0.1,0.5,0.9]),(32_000,[0.1,0.5,0.9]),(128_000,[0.1,0.5,0.9]),
      (256_000,[0.1,0.5,0.9]),(500_000,[0.1,0.5,0.9])]
rows=[]
print(f"{'target':>9} {'depth':>6} {'prompt_tok':>11} {'hit':>4} {'inans':>6} {'secs':>7}  answer")
for target,depths in PLAN:
    for d in depths:
        hay=build(target,d)
        p={"model":MODEL,"messages":[
            {"role":"user","content":hay+"\n\n"+Q}],
            "max_tokens":400,"temperature":0.0}
        t0=time.time()
        try:
            r=post(URL,p); dt=time.time()-t0; m=r["choices"][0]["message"]
            ans=(m.get("content") or "").strip(); rz=(m.get("reasoning_content") or "").strip()
            pt=r.get("usage",{}).get("prompt_tokens")
            in_ans=NUM in ans; hit=in_ans or (NUM in rz)
            shown=ans if ans else "[reasoning] "+rz
            rows.append((target,d,pt,hit,in_ans,dt,shown[:46]))
            print(f"{target:>9} {d:>6.2f} {str(pt):>11} {'YES' if hit else 'no':>4} {'yes' if in_ans else 'no':>6} {dt:>7.1f}  {shown[:46]!r}")
        except Exception as e:
            rows.append((target,d,None,None,None,time.time()-t0,f"ERR {e}"))
            print(f"{target:>9} {d:>6.2f} {'--':>11} {'ERR':>4} {'--':>6} {time.time()-t0:>7.1f}  {str(e)[:60]!r}")
        sys.stdout.flush()
print("\n# SUMMARY (markdown)")
print("| target | depth | prompt_tok | retrieved (in reasoning or answer) | echoed in final answer |")
print("|---|---|---|---|---|")
for (t,d,pt,hit,ia,dt,a) in rows:
    print(f"| {t:,} | {int(d*100)}% | {pt or '—'} | {'✅' if hit else '❌'} | {'✅' if ia else '❌'} |")
