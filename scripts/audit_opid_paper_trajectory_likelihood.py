#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, random, urllib.request
from pathlib import Path
import torch, transformers
from audit_gold_conditioned_teacher import load_jsonl
from audit_teacher_correction_retrieval import bootstrap_ci
from audit_think_search_teacher import action_parts, load_tree_metadata, score
from search_r1.llm_agent.self_opd import _opid_analyzer_prompt, _validate_opid_skill

def request(url,prompt):
    req=urllib.request.Request(url.rstrip('/')+'/generate',data=json.dumps({'prompt':prompt}).encode(),
        headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=120) as r: return json.loads(r.read())['skill']

def collect(records,meta,max_trajectories,min_gap,max_skills,min_success,min_failure):
    out=[]
    for rec in records:
        m=meta.get(str(rec.get('tree_uid')))
        if not m or not m.get('prompt'): continue
        nodes={n['node_uid']:n for n in rec['nodes']}
        for selected in rec.get('selected_leaves',[]):
            leaf=nodes.get(selected['node_uid'])
            if not leaf: continue
            path=[]; cur=leaf
            while cur and not cur.get('is_root'):
                path.append(cur); cur=nodes.get(cur.get('parent_node_uid'))
            path.reverse()
            steps=[]; candidates=[]
            for i,node in enumerate(path):
                parent=nodes[node['parent_node_uid']]
                prefix=parent.get('response') or ''
                full=node.get('response') or ''
                edge=full[len(prefix):] if full.startswith(prefix) else ''
                parts=action_parts(edge)
                if not parts: continue
                vals=[float(nodes[u].get('node_value',0)) for u in parent.get('child_node_uids',[]) if u in nodes]
                if len(vals)>=2 and max(vals)-min(vals)>=min_gap: candidates.append(i)
                steps.append({'index':i,'prefix':m['prompt']+prefix,'parts':parts,'edge':edge})
            if not steps: continue
            outcome=float(selected.get('original_score',leaf.get('node_value',0)))>=0.8
            formatted='\n'.join(f"[Step {x['index']}]\n{x['edge']}" for x in steps)
            try:
                sk=request(args.analyzer_url,_opid_analyzer_prompt(m['prompt'],outcome,candidates,formatted,max_skills))
            except Exception:
                continue
            if not _validate_opid_skill(sk,candidates): continue
            out.append({'tree_uid':rec['tree_uid'],'outcome':outcome,'steps':steps,'skill':sk})
            successes=sum(t['outcome'] for t in out); failures=len(out)-successes
            if len(out)>=max_trajectories or (successes>=min_success and failures>=min_failure): return out
    return out

def sheet(t,step):
    sk=t['skill']['step_skills'].get(str(step),t['skill']['episode_skill'])
    return '\n<information>Procedural guidance: '+sk.strip()+'</information>\n'

def mean(x):return sum(x)/len(x) if x else 0.
def main():
    global args
    ap=argparse.ArgumentParser();ap.add_argument('--trees',nargs='+',required=True);ap.add_argument('--selected',nargs='+',required=True)
    ap.add_argument('--model',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--analyzer-url',default='http://127.0.0.1:8128')
    ap.add_argument('--parquet',default='data/multihopqa_search_mixed_402020_20260830/train.parquet');ap.add_argument('--max-trajectories',type=int,default=24)
    ap.add_argument('--min-gap',type=float,default=.1);ap.add_argument('--max-skills',type=int,default=3);ap.add_argument('--max-context',type=int,default=2048)
    ap.add_argument('--min-success',type=int,default=32);ap.add_argument('--min-failure',type=int,default=32)
    ap.add_argument('--batch-size',type=int,default=2);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    tok=transformers.AutoTokenizer.from_pretrained(args.model);tok.pad_token_id=tok.pad_token_id or tok.eos_token_id
    rec=[r for pat in args.trees for p in glob.glob(pat) for r in load_jsonl(Path(p))]
    meta=load_tree_metadata(args.selected,args.parquet,tok);traj=collect(rec,meta,args.max_trajectories,args.min_gap,args.max_skills,args.min_success,args.min_failure)
    if len(traj)<8:raise SystemExit(f'only {len(traj)} trajectories')
    rng=random.Random(20260927);perm=list(range(len(traj)));rng.shuffle(perm)
    if any(i==p for i,p in enumerate(perm)):perm=perm[1:]+perm[:1]
    items=[];refs=[]
    for arm in ('plain','real','shuffle'):
      pairs=[]
      for i,t in enumerate(traj):
       donor=t if arm=='real' else traj[perm[i]]
       for st in t['steps']:
        extra='' if arm=='plain' else sheet(donor,st['index'])
        pairs.append((st['prefix']+extra,st['parts']));refs.append((arm,i,t['outcome'],st['index']))
      vals=score(pairs,model,tok,args.device,args.batch_size,args.max_context)
      for ref,val in zip(refs[-len(pairs):],vals):items.append({'arm':ref[0],'trajectory':ref[1],'success':ref[2],'step':ref[3],'score':val})
    summary={}
    for span in ('think','search','all'):
      summary[span]={}
      for outcome,label in ((True,'success'),(False,'failure')):
       deltas={}
       for arm in ('real','shuffle'):
        xs=[]
        for i in range(len(traj)):
         if traj[i]['outcome']!=outcome:continue
         a=[r['score'][span] for r in items if r['arm']==arm and r['trajectory']==i and r['score']]
         b=[r['score'][span] for r in items if r['arm']=='plain' and r['trajectory']==i and r['score']]
         if a and b:xs.append(mean(a)-mean(b))
        deltas[arm]={'n':len(xs),'mean':mean(xs),'ci95':bootstrap_ci(xs,seed=17) if xs else [0,0],
          'directional_fraction':mean([x>0 for x in xs]) if outcome else mean([x<0 for x in xs])}
       summary[span][label]=deltas
    s=summary['all']['success'];f=summary['all']['failure']
    gate={'pass':s['real']['n']>=args.min_success and f['real']['n']>=args.min_failure and s['real']['mean']>0 and f['real']['mean']<0
      and s['real']['mean']>s['shuffle']['mean'] and f['real']['mean']<f['shuffle']['mean'],
      'criteria':f'n_success>={args.min_success},n_failure>={args.min_failure}; success delta>0; failure delta<0; both directions beat shuffled'}
    out={'trajectory_count':len(traj),'gate':gate,'summary':summary,'trajectories':traj,'scores':items}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'trajectory_count':len(traj),'gate':gate,'summary':summary},indent=2))
if __name__=='__main__':
 import sys
 model_path=sys.argv[sys.argv.index('--model')+1];device=sys.argv[sys.argv.index('--device')+1] if '--device' in sys.argv else 'cuda:0'
 model=transformers.AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
 main()
