#!/usr/bin/env python3
"""Audit whether DeepSeek contrast skills raise good-action likelihood and margin."""
from __future__ import annotations
import argparse, glob, json, random, urllib.request
from pathlib import Path
from typing import Any
import torch, transformers
from audit_gold_conditioned_teacher import load_jsonl
from audit_teacher_correction_retrieval import anchored_prefix, bootstrap_ci
from audit_think_search_teacher import collect_events, load_tree_metadata, score
def norm_words(s): return set(''.join(c.lower() if c.isalnum() else ' ' for c in s).split())
def action_jaccard(a,b):
    x,y=norm_words(' '.join(a)),norm_words(' '.join(b)); return len(x&y)/max(1,len(x|y))
def prompt(state,good,bad,evidence):
    return f'''Analyze the following agent episode and return ONLY valid JSON.

You need to complete all three fields:
1. Write a concise episode_summary.
2. Write one episode_skill that extracts the successful trajectory into a workflow: the core decision rule and action ordering that made this trajectory work.
3. Provide concise, action-oriented decision guidance for the single critical step 0 as an entry in step_skills; use the full episode to infer the guidance, but phrase it as advice the policy can act on at that step.

Important constraints:
- Step indexing is 0-based.
- Use the task description together with the episode context to judge progress.
- The step_skills value must be one short imperative sentence for the policy at that step.
- Write step_skills as policy-facing guidance, not as retrospective explanation.
- Return only these top-level fields: episode_summary, episode_skill, step_skills.
- Do not reveal the final answer or mention scores, branches, hindsight, or private evidence.

Return format:
{{"episode_summary":"string","episode_skill":"string","step_skills":{{"0":"skill for step 0"}}}}

Episode context:
- Task description and interaction history before the critical step: {state[-9000:]}
- episode_success: true
- Candidate step indices: [0]
- Interaction trajectory:
[Step 0]
Observation: The visible state above.
Reasoning: {good[0]}
Action: <search>{good[1]}</search>
[Following environment feedback]
{evidence[:1800]}
'''
def request(url,p):
    req=urllib.request.Request(url.rstrip('/')+'/generate',data=json.dumps({'prompt':p}).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=120) as r: return json.loads(r.read())['skill']
def sheet(skill):
    return '\n<information>Procedural guidance: ' + skill['step_skills']['0'].strip() + '</information>\n'

def valid_skill(skill):
    return (isinstance(skill,dict) and set(skill)=={'episode_summary','episode_skill','step_skills'}
            and isinstance(skill['step_skills'],dict) and set(skill['step_skills'])=={'0'}
            and all(isinstance(skill[k],str) and skill[k].strip()
                    for k in ('episode_summary','episode_skill'))
            and isinstance(skill['step_skills']['0'],str)
            and 2 <= len(skill['step_skills']['0'].split()) <= 24)
def mean(xs): return sum(xs)/len(xs) if xs else 0.0
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--trees',nargs='+',required=True); ap.add_argument('--selected',nargs='+',required=True)
    ap.add_argument('--model',required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--analyzer-url',default='http://127.0.0.1:8128')
    ap.add_argument('--parquet',default='data/multihopqa_search_mixed_402020_20260830/train.parquet'); ap.add_argument('--max-events',type=int,default=24)
    ap.add_argument('--min-value-gap',type=float,default=.1); ap.add_argument('--max-jaccard',type=float,default=.8); ap.add_argument('--max-context',type=int,default=2048)
    ap.add_argument('--batch-size',type=int,default=2); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--seed',type=int,default=20260927); args=ap.parse_args()
    tok=transformers.AutoTokenizer.from_pretrained(args.model); tok.pad_token_id=tok.pad_token_id or tok.eos_token_id
    tree_paths=sorted(p for pat in args.trees for p in glob.glob(pat)); selected_paths=sorted(p for pat in args.selected for p in glob.glob(pat))
    records=[r for p in tree_paths for r in load_jsonl(Path(p))]; meta=load_tree_metadata(selected_paths,args.parquet,tok)
    events=[]
    for e in collect_events(records,args.min_value_gap,1800):
        m=meta.get(e['tree_uid']);
        if not m or not m.get('prompt') or action_jaccard(e['good'],e['bad'])>args.max_jaccard: continue
        state=m['prompt']+e['prefix']; answer=m.get('answer','')
        skill=request(args.analyzer_url,prompt(state,e['good'],e['bad'],e['evidence']))
        if not valid_skill(skill): continue
        e.update(state=state,answer=answer,skill=skill)
        e['prefix']=anchored_prefix(tok,m['prompt'],e['prefix'],max(256,args.max_context-500)); events.append(e)
        if len(events)>=args.max_events: break
    if len(events)<8: raise SystemExit(f'only {len(events)} valid diverse events')
    rng=random.Random(args.seed); perm=list(range(len(events))); rng.shuffle(perm)
    if any(perm[i]==i for i in range(len(events))): perm=perm[1:]+perm[:1]
    rows=[]
    for arm in ('plain','real','shuffle'):
        contexts=[]
        for i,e in enumerate(events):
            s='' if arm=='plain' else sheet(e['skill'] if arm=='real' else events[perm[i]]['skill'])
            contexts.append(e['prefix']+s)
        gs=score([(contexts[i],e['good']) for i,e in enumerate(events)],None if False else model,tok,args.device,args.batch_size,args.max_context)
        bs=score([(contexts[i],e['bad']) for i,e in enumerate(events)],model,tok,args.device,args.batch_size,args.max_context)
        if not rows: rows=[{'event':i,'tree_uid':e['tree_uid'],'parent_uid':e['parent_uid'],'skill':e['skill'],'scores':{}} for i,e in enumerate(events)]
        for i,(g,b) in enumerate(zip(gs,bs)): rows[i]['scores'][arm]={'good':g,'bad':b}
    complete=[r for r in rows if all(r['scores'][a]['good'] and r['scores'][a]['bad'] for a in ('plain','real','shuffle'))]
    def vals(span,arm,kind):
        out=[]
        for r in complete:
            p=r['scores']['plain']; a=r['scores'][arm]
            if kind=='good_lift': out.append(a['good'][span]-p['good'][span])
            else: out.append((a['good'][span]-a['bad'][span])-(p['good'][span]-p['bad'][span]))
        return out
    summary={}
    for span in ('think','search','all'):
        summary[span]={}
        for arm in ('real','shuffle'):
            gl=vals(span,arm,'good_lift'); ml=vals(span,arm,'margin_lift')
            summary[span][arm]={'good_lift_mean':mean(gl),'good_lift_ci95':bootstrap_ci(gl,seed=11),'good_lift_positive_fraction':mean([x>0 for x in gl]),'margin_lift_mean':mean(ml),'margin_lift_ci95':bootstrap_ci(ml,seed=13)}
    a=summary['all']['real']; s=summary['all']['shuffle']
    gate={'pass':len(complete)>=12 and a['good_lift_mean']>0 and a['margin_lift_mean']>0 and a['good_lift_positive_fraction']>=.6 and a['good_lift_mean']>s['good_lift_mean'] and a['margin_lift_mean']>s['margin_lift_mean'], 'criteria':'n>=12; real good lift>0; real margin lift>0; positive fraction>=0.6; both lifts beat shuffled skill'}
    out={'events_generated':len(events),'events_complete':len(complete),'gate':gate,'summary':summary,'rows':complete}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('events_generated','events_complete','gate','summary')},ensure_ascii=False,indent=2))
if __name__=='__main__':
    # Keep model global so score calls share one loaded checkpoint.
    args_model=None
    # argparse is parsed in main; load lazily by intercepting argv model value.
    import sys
    model_path=sys.argv[sys.argv.index('--model')+1]
    device=sys.argv[sys.argv.index('--device')+1] if '--device' in sys.argv else 'cuda:0'
    model=transformers.AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    main()
