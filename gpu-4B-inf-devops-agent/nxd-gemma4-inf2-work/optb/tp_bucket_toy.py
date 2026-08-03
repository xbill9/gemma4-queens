"""Minimal TP=2 + 2-bucket + shared-aliased-state toy to learn the bucketing plumbing.
Two buckets (seq=4 'prefill', seq=1 'decode') share ONE tiny ColumnParallelLinear's weights
and ONE state buffer (aliased). Goal: TRACED + state persists across a prefill then decode call."""
import sys, os, types, multiprocessing
m=types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None; sys.modules["transformers.utils.fx"]=m
TP=2; MAX=16; H=8

def get_model(mode="pre"):
    import torch
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
    class M(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.lin=ColumnParallelLinear(H,H,bias=False,gather_output=True)
            s.state=torch.nn.Parameter(torch.zeros(1,MAX,H),requires_grad=False)   # shared, aliased
        def forward(s, x):
            # x:[1,seq,H]; seq is a compile-time constant per bucket.
            y=s.lin(x); seq=x.shape[1]
            new=torch.cat([y, s.state[:, seq:, :]], dim=1)   # write y into first `seq` rows of state
            return y.sum(dim=1), new                          # (logits[1,H], updated state[1,MAX,H])
    mm=M().eval()
    return mm, {mm.state: 1}                                  # alias: state param -> output#1

def kernel_factory():
    import torch
    def kernel(inputs: list[torch.Tensor]):
        x=inputs[0]
        idx=torch.zeros(1,dtype=torch.long) if x.size(1)>1 else torch.ones(1,dtype=torch.long)
        return inputs, idx
    return torch.jit.script(kernel)

def main():
    import torch, neuronx_distributed
    from torch_neuronx import BucketModelConfig
    torch.manual_seed(0)
    x4=torch.randn(1,4,H); x1=torch.randn(1,1,H); state0=torch.zeros(1,MAX,H)
    bc=BucketModelConfig(kernel_factory, shared_state_buffer=[state0],
                         func_kwargs=[{"mode":"pre"},{"mode":"dec"}])
    print("tracing bucketed toy ...", flush=True)
    model=neuronx_distributed.trace.parallel_model_trace(
        get_model, [(x4,),(x1,)], bucket_config=bc, tp_degree=TP,
        inline_weights_to_neff=False, compiler_args=["--model-type","transformer"])
    print("TRACED", flush=True)
    o4=model(x4); print("prefill out type:", type(o4), flush=True)
    o1=model(x1); print("decode  out type:", type(o1), flush=True)
    print("prefill:", o4 if not isinstance(o4,(list,tuple)) else [t.shape for t in o4], flush=True)
    print("decode :", o1 if not isinstance(o1,(list,tuple)) else [t.shape for t in o1], flush=True)
    print("TOY_OK", flush=True)

if __name__=="__main__":
    if os.environ.get("_TP_CHILD")!="1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn",force=True); main()
