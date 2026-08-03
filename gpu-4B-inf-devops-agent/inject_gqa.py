import sys

def patch_server():
    server_path = "server.py"
    with open(server_path, "r") as f:
        code = f.read()

    gqa_patch = """
    # 8. Patch GQA for missing k_proj/v_proj in Gemma-4 global layers
    gqa_path = "/opt/conda/lib/python3.12/site-packages/neuronx_distributed_inference/modules/attention/gqa.py"
    if os.path.exists(gqa_path):
        with open(gqa_path, "r") as f:
            gqa_code = f.read()
        target = 'weight = model_state_dict[f"{prefix}.{layer_name}.weight"]'
        replacement = '''weight = model_state_dict.get(f"{prefix}.{layer_name}.weight")
            if weight is None:
                print(f"WARNING: {prefix}.{layer_name}.weight missing. Injecting zeros of shape {layer.weight.shape}!")
                weight = torch.zeros(layer.weight.shape, dtype=torch.bfloat16)'''
        if target in gqa_code:
            gqa_code = gqa_code.replace(target, replacement)
            with open(gqa_path, "w") as f:
                f.write(gqa_code)
            print("Patched gqa.py successfully for missing k_proj weights!")
"""

    insert_idx = code.find("# 7. Patch attention_base.py")
    if insert_idx != -1:
        code = code[:insert_idx] + gqa_patch + "\n" + code[insert_idx:]
        with open(server_path, "w") as f:
            f.write(code)
        print("Successfully injected gqa.py patch into server.py!")
    else:
        print("Could not find insertion point!")

if __name__ == "__main__":
    patch_server()
