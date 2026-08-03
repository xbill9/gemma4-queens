import os
import re

gqa_path = "/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/gqa.py"
if os.path.exists(gqa_path):
    with open(gqa_path, "r") as f:
        gqa_code = f.read()
    
    pattern = r"([ \t]*)weight = model_state_dict\[f\"\{prefix\}\.\{layer_name\}\.weight\"\]"
    def replacer(match):
        indent = match.group(1)
        return (
            f'{indent}weight = model_state_dict.get(f"{{prefix}}.{{layer_name}}.weight")\n'
            f'{indent}if weight is None:\n'
            f'{indent}    import torch\n'
            f'{indent}    print(f"WARNING: {{prefix}}.{{layer_name}}.weight missing. Injecting zeros of shape {{layer.weight.shape}}!")\n'
            f'{indent}    weight = torch.zeros(layer.weight.shape, dtype=torch.bfloat16)'
        )
        
    if re.search(pattern, gqa_code):
        gqa_code = re.sub(pattern, replacer, gqa_code)
        with open(gqa_path, "w") as f:
            f.write(gqa_code)
        print("Patched gqa.py successfully for missing k_proj weights!")
    else:
        print("Target string not found in gqa.py!")
