#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import torch,transformers
from audit_teacher_correction_retrieval import bootstrap_ci
from audit_think_search_teacher import score

def mean(x):return sum(x)/len(x) if x else 0.
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--model',required=True)
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--batch-size',type=int,default=3);ap.add_argument('--max-context',type=int,default=2048);args=ap.parse_args()
 d=json.loads(args.input.read_text());ts=d['trajectories'];plain={(r['trajectory'],r['step']):r['score'] for r in d['scores'] if r['arm']=='plain' and r['score']}
 tok=transformers.AutoTokenizer.from_pretrained(args.model);tok.pad_token_id=tok.pad_token_id or tok.eos_token_id
 rng=random.Random(20260927);perm=list(range(len(ts)));rng.shuffle(perm)
 if any(i==v for i,v in enumerate(perm)):perm=perm[1:]+perm[:1]
 rows=[]
 for arm in ('real','shuffle'):
  pairs=[];refs=[]
  for i,t in enumerate(ts):
   sk=(t if arm=='real' else ts[perm[i]])['skill']['episode_skill'].strip()
   sheet='\n<information>Procedural guidance: '+sk+'</information>\n'
   for st in t['steps']:
    pairs.append((st['prefix']+sheet,st['parts']));refs.append((i,st['index']))
  vals=score(pairs,model,tok,args.device,args.batch_size,args.max_context)
  for (i,step),v in zip(refs,vals):
   if v and (i,step) in plain:rows.append({'arm':arm,'trajectory':i,'step':step,'success':ts[i]['outcome'],'score':v})
 summary={}
 for success,label in ((True,'success'),(False,'failure')):
  by={}
  for arm in ('real','shuffle'):
   xs=[]
   for i,t in enumerate(ts):
    if t['outcome']!=success:continue
    av=[r['score']['search'] for r in rows if r['arm']==arm and r['trajectory']==i]
    pv=[plain[i,st['index']]['search'] for st in t['steps'] if (i,st['index']) in plain]
    if av and pv:xs.append(mean(av)-mean(pv))
   by[arm]=xs
  diff=[a-b for a,b in zip(by['real'],by['shuffle'])]
  summary[label]={arm:{'n':len(xs),'mean':mean(xs),'ci95':bootstrap_ci(xs,seed=41),'negative_fraction':mean([x<0 for x in xs])} for arm,xs in by.items()}
  summary[label]['real_minus_shuffle']={'mean':mean(diff),'ci95':bootstrap_ci(diff,seed=43)}
 f=summary['failure']
 gate={'pass':f['real']['mean']<0 and f['real_minus_shuffle']['ci95'][1]<0 and f['real']['negative_fraction']>=.6,
       'criteria':'failure real delta<0; paired real-shuffle CI95 upper<0; >=60% failure trajectories negative'}
 out={'gate':gate,'summary':summary,'rows':rows};args.output.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'gate':gate,'summary':summary},indent=2))
if __name__=='__main__':
 import sys
 mp=sys.argv[sys.argv.index('--model')+1];dev=sys.argv[sys.argv.index('--device')+1] if '--device' in sys.argv else 'cuda:0'
 model=transformers.AutoModelForCausalLM.from_pretrained(mp,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(dev).eval();main()
