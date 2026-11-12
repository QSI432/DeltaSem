import json
import itertools
import numpy as np
from pathlib import Path
from scipy.stats import hypergeom
from deltasem.certificates import hypergeom_count_interval, transport, safe_midpoint

# Exact probability computation, not Monte Carlo: every population K, n, N<=24.
alpha=0.05
max_failure=0.0
coverage_cases=0
for N in range(1,25):
    for n in range(1,N+1):
        cis={x:hypergeom_count_interval(N,n,x,alpha) for x in range(n+1)}
        for K in range(N+1):
            fail=sum(float(hypergeom.pmf(x,N,K,n)) for x,(L,U) in cis.items() if not L<=K<=U)
            assert fail <= alpha + 1e-10, (N,n,K,fail)
            max_failure=max(max_failure,fail)
            coverage_cases+=1

rng=np.random.default_rng(987312)
transport_cases=0
for N in range(1,9):
    w=np.arange(1,N+1,dtype=float)
    for bits in itertools.product([0,1],repeat=N):
        y=np.array(bits,dtype=float)
        total=float(w@y)
        L=max(0.0,total-float(rng.uniform(0,5)))
        U=min(float(w.sum()),total+float(rng.uniform(0,5)))
        for _ in range(10):
            mode=rng.integers(0,3,N)  # 0 retained, 1 unresolved deletion, 2 resolved promotion
            new_w=rng.uniform(0,5,size=3)
            new_y=rng.integers(0,2,3)
            wd=float(w[mode==1].sum())
            vz=float((w*y)[mode==2].sum())
            wa=float(new_w.sum())
            mass=float(w[mode==0].sum()+wa)
            truth=float((w*y)[mode==0].sum()+new_w@new_y)
            l,u=transport(L,U,wd,wa,vz,mass)
            assert l-1e-9<=truth<=u+1e-9
            transport_cases+=1

contract_cases=0
for _ in range(10000):
    A=float(rng.uniform(0,100))
    L=max(0,A-float(rng.uniform(0,5)))
    U=A+float(rng.uniform(0,5))
    eps=float(rng.uniform(.01,.2))
    floor=float(rng.uniform(1,20))
    estimate=safe_midpoint(L,U,eps,floor)
    if estimate is not None:
        assert abs(estimate-A)<=eps*max(A,floor)+1e-10
    contract_cases+=1
report={
  'scope':'Mathematical building-block validation only. No empirical LLM or serving measurements.',
  'hypergeom_exact_coverage_cases':coverage_cases,
  'nominal_alpha':alpha,
  'worst_exact_failure_probability':max_failure,
  'pathwise_transfer_cases':transport_cases,
  'mixed_error_contract_cases':contract_cases,
  'status':'passed'
}
Path(__file__).with_name('validation_results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
