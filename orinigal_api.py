
########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
# RWKV语言模型，GitHub地址：https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print('\nChatRWKV https://github.com/BlinkDL/ChatRWKV\n')

import os, sys, torch
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# current_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(f'{current_path}/rwkv_pip_package/src')

# Tune these below (test True/False for all of them) to find the fastest setting:
# 下面这些参数可以调整，以找到最快的设置：
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

########################################################################################################
#
# Use '/' in model path, instead of '\'. Use ctx4096 models if you need long ctx.
# 模型路径中使用“/”而不是“\”，如果需要较长的上下文，请使用ctx4096模型。
#
# fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# fp32 = good for CPU
# bf16 = worse accuracy, supports CPU
# xxxi8 (example: fp16i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
#
# Read https://pypi.org/project/rwkv/ for Strategy Guide
# 请阅读https://pypi.org/project/rwkv/以获取策略指南。
########################################################################################################
# set these before import RWKV
# 在导入RWKV之前设置这些参数。
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

from rwkv.model import RWKV # pip install rwkv
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cuda fp16')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cuda fp16i8')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cuda fp16i8 *6 -> cuda fp16 *0+ -> cpu fp32 *1')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cpu fp32')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cpu fp32 *3 -> cuda fp16 *6+')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cpu fp32')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16 *8 -> cpu fp32')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda:0 fp16 -> cuda:1 fp16 -> cpu fp32 *1')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16 *6+')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230213-8019', strategy='cuda fp16 *0+ -> cpu fp32 *1')
# model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096', strategy='cuda:0 fp16 *25 -> cuda:1 fp16')

# out, state = model.forward([187], None)
# print(out.detach().cpu().numpy())

out, state = model.forward([187, 510, 1563, 310, 247], None)
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())                   # same result as above

# print('\n')
# exit(0)
from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "20B_tokenizer.json")

ctx = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
print(ctx, end='')

def my_print(s):
    print(s, end='', flush=True)

# For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
# https://platform.openai.com/docs/api-reference/parameter-details
# alpha_frequency和alpha_presence参数的详细信息请参见“Frequency and presence penalties”页面。
args = PIPELINE_ARGS(temperature = 1.0, max_tokens = 50, top_p = 0.95, frequency_penalty = 0.0, presence_penalty = 0.0, stop=["\n"])
# args = PIPELINE_ARGS(temperature = 1.0, max_tokens = 50, top_p = 0.95, frequency_penalty = 0.5, presence_penalty = 0.5, stop=["\n"])

while True:
    try:
        text = input()
        if text == 'quit':
            break
        ctx += text + '\n'
        prompt = ctx.replace('\n', ' ')
        out = pipeline(prompt, **args)
        my_print(out)
        ctx += out + '\n'
    except:
        import traceback
        traceback.print_exc()