#!/bin/bash
# LoRA train script by @Akegarasu modify by @bdsqlsz
# Train Model | 训练模式
model="hunyuan_db" #lora、db、sdxl_lora、sdxl_db、sdxl_cn3l、contralnet、stable_cascade_db、stable_cascade_lora、hunyuan_lora、hunyuan_db、sd3_db

# Train data path | 设置训练用模型、图片
pretrained_model="./Stable-diffusion/HunYuanDiT_fp16" # base model path | 底模路径
is_v2_model=0                                                # SD2.0 model | SD2.0模型 2.0模型下 clip_skip 默认无效
v_parameterization=1                                         # parameterization | 参数化 v2 非512基础分辨率版本必须使用。
vae=""
train_data_dir="./train/qinglong/train"                      # train dataset path | 训练数据集路径
reg_data_dir=""                                        # directory for regularization images | 正则化数据集路径，默认不使用正则化图像。
training_comment="this_LoRA_model_credit_from_bdsqlsz" # training_comment | 训练介绍，可以写作者名或者使用触发关键词

#SD3
clip_l="./clip/clip_l.safetensors"
clip_g="./clip/clip_g.safetensors"
t5xxl="./clip/t5xxl_fp16.safetensors"
t5xxl_device=""    #默认cuda，显存不够可改为CPU，但是很慢
t5xxl_dtype="bf16" #目前支持fp32、fp16、bf16
text_encoder_batch_size=12
num_last_block_to_freeze=240

# Train related params | 训练相关参数
resolution="1024,1024" # image resolution w,h. 图片分辨率，宽,高。支持非正方形，但必须是 64 倍数。
batch_size=4          # batch size
vae_batch_size=4       #vae初始化转换图片批处理大小，2-4。大了可以让一开始处理图片更快
max_train_epoches=32   # max train epoches | 最大训练 epoch
save_every_n_epochs=2  # save every n epochs | 每 N 个 epoch 保存一次

gradient_checkpointing=1      #梯度检查，开启后可节约显存，但是速度变慢
gradient_accumulation_steps=0 # 梯度累加数量，变相放大batchsize的倍数

network_dim=32   # network dim | 常用 4~128，不是越大越好
network_alpha=16 # network alpha | 常用与 network_dim 相同的值或者采用较小的值，如 network_dim的一半 防止下溢。默认值为 1，使用较小的 alpha 需要提升学习率。

#dropout | 抛出(目前和lycoris不兼容，请使用lycoris自带dropout)
network_dropout="0"                # dropout 是机器学习中防止神经网络过拟合的技术，建议0.1~0.3
scale_weight_norms="1.0"           #配合 dropout 使用，最大范数约束，推荐1.0
rank_dropout="0"                   #lora模型独创，rank级别的dropout，推荐0.1~0.3，未测试过多
module_dropout="0"                 #lora模型独创，module级别的dropout(就是分层模块的)，推荐0.1~0.3，未测试过多
caption_dropout_every_n_epochs="0" #dropout caption
caption_dropout_rate="0"           #0~1
caption_tag_dropout_rate="0.1"     #0~1
max_grad_norm="1.0"

train_unet_only=0         # train U-Net only | 仅训练 U-Net，开启这个会牺牲效果大幅减少显存使用。6G显存可以开启
train_text_encoder_only=0 # train Text Encoder only | 仅训练 文本编码器

seed="1026" # reproducable seed | 设置跑测试用的种子，输入一个prompt和这个种子大概率得到训练图。可以用来试触发关键词

#噪声
noise_offset="0"                 # noise offset | 在训练中添加噪声偏移来改良生成非常暗或者非常亮的图像，如果启用，推荐参数为0.1
adaptive_noise_scale="0"         #adaptive noise scale | 自适应噪声偏移范围
noise_offset_random_strength=0   #0是关，1是开。噪声随机强度
multires_noise_iterations="0"    #多分辨率噪声扩散次数，推荐6-10,0禁用,和noise_offset冲突，只能开一个
multires_noise_discount="0"      #多分辨率噪声缩放倍数，推荐0.1-0.3,上面关掉的话禁用。
min_snr_gamma="0"                #最小信噪比伽马值，减少低step时loss值，让学习效果更好。推荐3-5，5对原模型几乎没有太多影响，3会改变最终结果。修改为0禁用。
weighted_captions=0              #权重打标，默认识别标签权重，语法同webui基础用法。例如(abc), [abc], (abc:1.23),但是不能再括号内加逗号，否则无法识别。
ip_noise_gamma="0"               #误差噪声添加，防止误差累计
ip_noise_gamma_random_strength=0 #0是关，1是开。误差噪声随机强度
debiased_estimation_loss=0       #0是关，1是开。信噪比噪声修正，minsnr高级版
immiscible_noise=0               #0是关，1是开。不可混合噪声，让噪声更加真实

#标签编辑
shuffle_caption=1           # 随机打乱tokens顺序，默认启用。修改为 0 禁用。
keep_tokens=1               # keep heading N tokens when shuffling caption tokens | 在随机打乱 tokens 时，保留前 N 个不变。
prior_loss_weight="1"       #正则化权重，0-1
secondary_separator=";;;"   #次要分隔符。被该分隔符分隔的部分将被视为一个token，并被洗牌和丢弃。然后由 caption_separator 取代。例如，如果指定 aaa;;bbb;;cc，它将被 aaa,bbb,cc 取代或一起丢弃。
keep_tokens_separator="|||" #批量保留不变，间隔符号
enable_wildcard=0           #通配符随机抽卡，格式参考 {aaa|bbb|ccc}
caption_prefix=""           #打标前缀，可以加入质量词如果底模需要，例如masterpiece, best quality,
caption_suffix=""           #打标后缀，可以加入相机镜头如果需要，例如full body等
alpha_mask=0                #是否使用透明蒙版检测

# Learning rate | 学习率
lr="2e-4"
unet_lr="8e-4"
text_encoder_lr="2e-5"
lr_scheduler="warmup_stable_decay" # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
# 新增cosine_with_min_lr(适合训练lora)、warmup_stable_decay(适合训练db)、inverse_sqrt
lr_warmup_steps=0               # warmup steps | 仅在 lr_scheduler 为 constant_with_warmup 时需要填写这个值
lr_decay_steps=30000            # decay steps | 学习率衰减步数，仅在 lr_scheduler 为warmup_stable_decay时 需要填写，一般是10%总步数
lr_scheduler_num_cycles=1       # cosine_with_restarts restart cycles | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时起效。
lr_scheduler_timescale=0        #times scale |时间缩放，仅在 lr_scheduler 为 inverse_sqrt 时需要填写这个值，默认同lr_warmup_steps
lr_scheduler_min_lr_ratio="0.1" #min lr ratio |最小学习率比率，仅在 lr_scheduler 为 cosine_with_min_lr、、warmup_stable_decay 时需要填写这个值，默认0

# Output settings | 输出设置
output_name="hunyuan_db"  # output model name | 模型保存名称
save_model_as="safetensors" # model save ext | 模型保存格式 ckpt, pt, safetensors
mixed_precision="fp16"      # 默认fp16,可选 "fp16", "bf16","no"
save_precision="fp16"       # 默认fp16,可选 "fp16", "bf16","fp32"
full_fp16=1                 # 半精度全部使用fp16
full_bf16=0                 # 半精度全部使用bf16
fp8_base=0                  # 实验性功能FP8训练
cache_latents=1             #缓存潜变量
cache_latents_to_disk=1     #开启缓存潜变量保存到磁盘，这样下次训练不用再次缓存转换，速度更快
no_half_vae=0               #禁止半精度，防止黑图。无法和mixed_precision混合精度共用。

#保存状态
save_state=1              # save training state | 保存训练状态 名称类似于 <output_name>-??????-state ?????? 表示 epoch 数
resume=""                 # resume from state | 从某个状态文件夹中恢复训练 需配合上方参数同时使用 由于规范文件限制 epoch 数和全局步数不会保存 即使恢复时它们也从 1 开始 与 network_weights 的具体实现操作并不一致
save_state_on_train_end=1 #只在训练结束最后保存训练状态

# wandb
wandb_api_key=""
log_tracker_name=$output_name

# Sample output | 出图
enable_sample=0                      #开启出图
sample_every_n_epochs=2              #每n个epoch出一次图
sample_prompts="./toml/qinglong.txt" #prompt文件路径
sample_sampler="euler_a"             #采样器 'ddim', 'pndm', 'heun', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'

# 其他设置
network_weights=""               # pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请填写 LoRA 模型路径。
enable_bucket=1                  # arb for diff wh | 分桶
min_bucket_reso=512              # arb min resolution | arb 最小分辨率
max_bucket_reso=1536             # arb max resolution | arb 最大分辨率
bucket_no_upscale=1              # arb no upscale | arb 小图不放大
persistent_data_loader_workers=1 # persistent dataloader workers | 容易爆内存，保留加载训练集的worker，减少每个 epoch 之间的停顿
clip_skip=1                      # clip skip | 玄学 SD1.5用 2
torch_compile=0                  #使用torch编译功能，需要版本大于2.1
dynamo_backend="aot_eager"       #"eager", "aot_eager", "inductor","aot_ts_nvfuser","nvprims_nvfuser","cudagraphs","aot_torchxla_trace_once"用于训练

# 优化器设置
#use_8bit_adam=1 # use 8bit adam optimizer | 使用 8bit adam 优化器节省显存，默认启用。部分 10 系老显卡无法使用，修改为 0 禁用。
#use_lion=0      # use lion optimizer | 使用 Lion 优化器
optimizer_type="AdamW8bit" # "adaFactor","AdamW8bit","Lion","DAdaptation",  推荐新优化器Lion。推荐学习率unetlr=lr=6e-5,tenclr=7e-6
# 新增优化器"Lion8bit"(速度更快，内存消耗更少)、"DAdaptAdaGrad"、"DAdaptAdan"(北大最新算法，效果待测)、"DAdaptSGD"
# 新增优化器 Sophia(2倍速1.7倍显存)、Prodigy天才优化器，可自适应Dylora
# 新增优化器 AdamWScheduleFree、SGDScheduleFree
# 新增优化器 StableAdamW、Ranger
d0="4e-7"             # d0 | prodigy的初始学习率 4e-7
fused_backward_pass=0 # use fused backward pass | 使用融合后的反向传播,训练大模型float32精度专用节约显存，必须优化器adafactor，gradient_accumulation_steps必须为1或者不开。

#gorkfast | 快速拟合
gradfilter_ema_alpha="0" #EMA的动量超参数 设置ema_alpha来激活gradfilter_ema，推荐0.98，为0则关闭
gradfilter_ema_lamb=2.0  #滤波器ema的放大因子超参数。

# lycoris 训练设置
enable_lycoris_train=0 # enable lycoris train | 启用 LoCon 训练 启用后 network_dim 和 network_alpha 应当选择较小的值，比如 2~16
conv_dim=0             # conv dim | 类似于 network_dim，推荐为 4
conv_alpha=0           # conv alpha | 类似于 network_alpha，可以采用与 conv_dim 一致或者更小的值
algo="lokr"            # algo参数，制定训练lycoris模型种类，包括lora(locon)、loha、IA3以及lokr、dylora 。5个可选
dropout="0"            #lycoris专用dropout
preset="attn-mlp"      #预设训练模块配置
#full: default preset, train all the layers in the UNet and CLIP|默认设置，训练所有Unet和Clip层
#full-lin: full but skip convolutional layers|跳过卷积层
#attn-mlp: train all the transformer block.|kohya配置，训练所有transformer模块
#attn-only：only attention layer will be trained, lot of papers only do training on attn layer.|只有注意力层会被训练，很多论文只对注意力层进行训练。
#unet-transformer-only： as same as kohya_ss/sd_scripts with disabled TE, or, attn-mlp preset with train_unet_only enabled.|和attn-mlp类似，但是关闭te训练
#unet-convblock-only： only ResBlock, UpSample, DownSample will be trained.|只训练卷积模块，包括res、上下采样模块
#./toml/example_lycoris.toml: 也可以直接使用外置配置文件，制定各个层和模块使用不同算法训练，需要输入位置文件路径，参考样例已添加。

factor=8     #只适用于lokr的因子，-1~8，8为全维度
block_size=4 #适用于dylora,分割块数单位，最小1也最慢。一般4、8、12、16这几个选
use_tucker=1 #适用于除 (IA)^3 和full
use_scalar=1 #根据不同算法，自动调整初始权重
train_norm=1 #归一化层
full_matrix=1

# dylora 训练设置
enable_dylora_train=0 # enable dylora train | 启用 LoCon 训练 启用后 network_dim 和 network_alpha 应当选择较小的值，比如 2~16
unit=4                #block size

#Lora_FA
enable_lora_fa=0 # 开启lora_fa，和lycoris、dylora冲突，只能开一个。

#oft
enable_oft=0 # 开启oft，和已上冲突，只能开一个。

#LORA_PLUS
enable_lora_plus=1
loraplus_lr_ratio=16
loraplus_unet_lr_ratio=16
loraplus_text_encoder_lr_ratio=4

# Merge lora and train | 差异提取法
base_weights=""
base_weights_multiplier="1.0"

# Block weights | 分层训练
enable_block_weights=0                         #开启分层训练
down_lr_weight="1,0.2,1,1,0.2,1,1,0.2,1,1,1,1" #12层，需要填写12个数字，0-1.也可以使用函数写法，支持sine, cosine, linear, reverse_linear, zeros，参考写法down_lr_weight=cosine+.25
mid_lr_weight="1"                              #1层，需要填写1个数字，其他同上。
up_lr_weight="1,1,1,1,1,1,1,1,1,1,1,1"         #12层，同上上。
block_lr_zero_threshold=0                      #如果分层权重不超过这个值，那么直接不训练。默认0。

enable_block_dim=0                                                                           #开启分块dim训练
block_dims="64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64"      #dim分块，25块
block_alphas="1,1,2,1,2,2,4,1,1,4,4,4,1,4,1,4,2,1,1,4,1,1,1,4,1"                             #alpha分块，25块
conv_block_dims="32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32" #convdim分块，25块
conv_block_alphas="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"                        #convalpha分块，25块

# SDXL
min_timestep="0"                     #最小时序，默认值0
max_timestep="1000"                  #最大时序，默认值1000
bucket_reso_steps="32"               #default 64,SDXL can use 32
cache_text_encoder_outputs=1         #开启缓存文本编码器，开启后减少显存使用。但是无法和shuffle共用
cache_text_encoder_outputs_to_disk=1 #开启缓存文本编码器到磁盘，开启后减少显存使用。但是无法和shuffle共用

#checkpoint train
no_token_padding=0           #不进行分词器填充
stop_text_encoder_training=0 #在N步后停止文本编码器训练
train_text_encoder=1         #训练文本编码器
learning_rate_te="5e-8"      #文本编码器学习率 SD1.5/SD2.1

#SDXL_db
diffuser_xformers=0      #开启diffuser的xformers
learning_rate_te1="5e-8" #文本编码器1学习率
learning_rate_te2="5e-8" #文本编码器2学习率

# block lr | SDXL_DB分层训练
enable_block_lr=0
block_lr="0,$lr,$lr,0,$lr,$lr,0,$lr,$lr,0,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,0"

#cn3l
conditioning_data_dir="" #条件图数据目录
cond_emb_dim=256         #条件图向量维度
masked_loss=0            #开启蒙版loss，对条件图处理，R通道255视为掩码mask，0视为无掩码

#multi_gpu | 多卡设置
multi_gpu=0                         #multi gpu | 多显卡训练开关，0关1开， 该参数仅限在显卡数 >= 2 使用
deepspeed=0                         #deepspeed | 使用deepspeed训练，0关1开， 该参数仅限在显卡数 >= 2 使用
zero_stage=2                        #zero stage | zero stage 0,1,2,3,阶段2用于训练 该参数仅限在显卡数 >= 2 使用
offload_optimizer_device=""      #offload optimizer device | 优化器放置设备，cpu或者nvme, 该参数仅限在显卡数 >= 2 使用
fp16_master_weights_and_gradients=0 #fp16 master weights and gradients | fp16主权重和梯度，0关1开， 该参数仅限在显卡数 >= 2 使用

ddp_timeout=1
ddp_gradient_as_bucket_view=1
ddp_static_graph=1

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
if [ -d "venv/bin/" ]; then
  source "venv/bin/activate"
fi
export HF_HOME="huggingface"
export TF_CPP_MIN_LOG_LEVEL=3
export HF_ENDPOINT="https://hf-mirror.com"

network_module="networks.lora"
extArgs=()
launchArgs=()
train_script="train_network"

if [ "$model" = "stable_cascade_lora" ]; then
  train_script="stable_cascade_train_c_network"
else
  train_script="train_network"
fi

if [[ $multi_gpu == 1 ]]; then
  launchArgs+=("--multi_gpu")
  launchArgs+=("--mixed_precision=$mixed_precision")
  if [[ $deepspeed == 1 ]]; then
    extArgs+=("--deepspeed")
    if [[ $zero_stage != 0 ]]; then
      extArgs+=("--zero_stage=$zero_stage")
    fi
    if [ $offload_optimizer_device ]; then
      extArgs+=("--offload_optimizer_device=$offload_optimizer_device")
    fi
    if [[ $fp16_master_weights_and_gradients == 1 ]]; then
      extArgs+=("--fp16_master_weights_and_gradients")
    fi
  fi
  if [ $ddp_timeout -ne 0 ]; then
    extArgs+=("--ddp_timeout")
  fi
  if [ $ddp_gradient_as_bucket_view -ne 0 ]; then
    extArgs+=("--ddp_gradient_as_bucket_view")
  fi
  if [ $ddp_static_graph -ne 0 ]; then
    extArgs+=("--ddp_static_graph")
  fi
fi

if [ $fused_backward_pass -ne 0 ]; then
  extArgs+=("--fused_backward_pass")
  gradient_accumulation_steps=0
  full_fp16=0
  full_bf16=0
  fp8_base=0
  save_precision="fp16"
fi

if [ $no_half_vae -ne 0 ]; then
  extArgs+=("--no_half_vae")
  mixed_precision=""
  full_bf16=0
  full_fp16=0
  fp8_base=0
fi

if [ $fp8_base -ne 0 ]; then
  extArgs+=("--fp8_base")
  full_fp16=1
  save_precision="fp16"
  full_bf16=0
fi

if [ $full_bf16 -ne 0 ]; then
  extArgs+=("--full_bf16")
  mixed_precision="bf16"
  full_fp16=0
elif [ $full_fp16 -ne 0 ]; then
  extArgs+=("--full_fp16")
  mixed_precision="fp16"
fi

if [[ $model != *lora ]]; then
  network_module=""
  if [[ $model == *db || $model == "controlnet" ]]; then
    network_dim=""
    train_script="train"
    if [[ $model == "db" ]]; then
      train_script=$train_script+"_db"
      if [[ $no_token_padding -ne 0 ]]; then extArgs+=("--no_token_padding"); fi
      if [[ $learning_rate_te != "0" ]]; then extArgs+=("--learning_rate_te=$learning_rate_te"); fi
      if [[ $stop_text_encoder_training -ne 0 ]]; then extArgs+=("--stop_text_encoder_training=$stop_text_encoder_training"); fi
    elif [[ $model == "sdxl_db" ]]; then
      if [[ $enable_block_lr -ne 0 ]]; then extArgs+=("--block_lr=$block_lr"); fi
      if [[ $diffuser_xformers -ne 0 ]]; then extArgs+=("--diffuser_xformers"); fi
      if [[ $train_text_encoder -ne 0 ]]; then
        extArgs+=("--train_text_encoder")
        if [[ $learning_rate_te1 != "0" ]]; then extArgs+=("--learning_rate_te1=$learning_rate_te1"); fi
        if [[ $learning_rate_te2 != "0" ]]; then extArgs+=("--learning_rate_te2=$learning_rate_te2"); fi
      fi
    fi
  fi
  network_alpha=""
  network_weights=""
  enable_block_weights=0
  enable_block_dim=0
  enable_lycoris_train=0
  enable_dylora_train=0
  enable_lora_fa=0
  enable_oft=0
  enable_lora_plus=0
  unet_lr=""
  text_encoder_lr=""
  train_unet_only=0
  train_text_encoder_only=0
  training_comment=""
  prior_loss_weight=1
  network_dropout="0"
fi

if [[ $model == sdxl* ]]; then
  train_script="sdxl_$train_script"
  if [ $cache_text_encoder_outputs -ne 0 ]; then
    extArgs+=("--cache_text_encoder_outputs")
    enable_bucket=0
    shuffle_caption=0
    caption_dropout_every_n_epochs="0"
    caption_dropout_rate="0"
    caption_tag_dropout_rate="0"
    if [ $cache_text_encoder_outputs_to_disk -ne 0 ]; then
      extArgs+=("--cache_text_encoder_outputs_to_disk")
    fi
  fi

  if [ $bucket_reso_steps != "64" ]; then
    extArgs+=("--bucket_reso_steps=$bucket_reso_steps")
  fi

  if [ $min_timestep != "0" ]; then
    extArgs+=("--min_timestep=$min_timestep")
  fi

  if [ $max_timestep != "1000" ]; then
    extArgs+=("--max_timestep=$max_timestep")
  fi
fi

if [[ $model == *cn3l || $model == "controlnet" ]]; then
  if [[ $model == "controlnet" ]]; then
    train_script="train_controlnet"
  else
    train_script="sdxl_train_control_net_lllite"
    if [ $cond_emb_dim -ne 0 ]; then
      extArgs+=("--cond_emb_dim=$cond_emb_dim")
    fi
  fi

  if [ $conditioning_data_dir ]; then
    extArgs+=("--conditioning_data_dir=$conditioning_data_dir")
  fi
  if [ $masked_loss -ne 0 ]; then
    extArgs+=("--masked_loss")
  fi
fi

if [[ $model == sd3* ]]; then
  train_script="sd3_$train_script"

  if [ -n "$clip_l" ]; then
    extArgs+=("--clip_l=$clip_l")
  fi

  if [ -n "$clip_g" ]; then
    extArgs+=("--clip_g=$clip_g")
  fi

  if [ -n "$t5xxl" ]; then
    extArgs+=("--t5xxl=$t5xxl")
  fi

  if [ -n "$t5xxl_device" ]; then
    extArgs+=("--t5xxl_device=$t5xxl_device")
  fi

  if [ -n "$t5xxl_dtype" ]; then
    extArgs+=("--t5xxl_dtype=$t5xxl_dtype")
  fi

  if [ "$text_encoder_batch_size" -ne 0 ]; then
    extArgs+=("--text_encoder_batch_size=$text_encoder_batch_size")
  fi

  if [ "$cache_text_encoder_outputs" -ne 0 ]; then
    extArgs+=("--cache_text_encoder_outputs")
    shuffle_caption=0
    loraplus_text_encoder_lr_ratio=0
    caption_dropout_rate=0
    caption_tag_dropout_rate=0
    if [ $cache_text_encoder_outputs_to_disk -ne 0 ]; then
      extArgs+=("--cache_text_encoder_outputs_to_disk")
    fi
  fi

  if [ "$num_last_block_to_freeze" -ne 0 ]; then
    extArgs+=("--num_last_block_to_freeze=$num_last_block_to_freeze")
  fi

  extArgs+=("--sdpa")
else
  extArgs+=("--xformers")
fi

if [ $enable_lycoris_train == 1 ]; then
  network_module="lycoris.kohya"
  lycoris_network_args=" algo=$algo"
  if [[ $use_scalar -ne 0 ]]; then
    lycoris_network_args+=(" use_scalar=True")
  fi
  if [[ $train_norm -ne 0 ]]; then
    lycoris_network_args+=(" train_norm=True")
  fi
  if [[ $algo != "ia3" ]]; then
    if [[ $algo != "full" ]]; then
      if [[ $conv_dim -ne 0 ]]; then
        lycoris_network_args+=(" conv_dim=$conv_dim conv_alpha=$conv_alpha")
      fi
      if [[ $use_tucker -ne 0 ]]; then
        lycoris_network_args+=(" use_tucker=True")
      fi
    fi
    lycoris_network_args+=(" preset=$preset")
  fi
  if [[ $algo == "locon" && dropout != "0" ]]; then
    lycoris_network_args+=(" dropout=$dropout")
  fi
  if [[ $algo == "lokr" ]]; then
    lycoris_network_args+=(" factor=$factor")
    lycoris_network_args+=(" full_matrix=True")
  elif [[ $algo == "dylora" ]]; then
    lycoris_network_args+=(" block_size=$block_size")
  fi
  extArgs+=("--network_args $lycoris_network_args")

elif [ $enable_dylora_train == 1 ]; then
  network_module="networks.dylora"
  extArgs+=("--network_args unit=$unit")
  if [[ $module_dropout != "0" ]]; then
    extArgs+=("module_dropout=$module_dropout")
  fi

elif [ $enable_lora_fa -ne 0 ]; then
  network_module="networks.lora_fa"

elif [ $enable_oft -ne 0 ]; then
  network_module="networks.oft"

elif [ $enable_block_weights == 1 ]; then
  extArgs+=("--network_args down_lr_weight=$down_lr_weight mid_lr_weight=$mid_lr_weight up_lr_weight=$up_lr_weight block_lr_zero_threshold=$block_lr_zero_threshold")
  if [ $enable_block_dim == 1 ]; then
    extArgs+=("block_dims=$block_dims block_alphas=$block_alphas")
    if [ $conv_block_dims ]; then
      extArgs+=("conv_block_dims=$conv_block_dims conv_block_alphas=$conv_block_alphas")
    fi
  fi
fi

if [ $network_module ]; then extArgs+=("--network_module=$network_module"); fi

if [ "$enable_lora_plus" -eq 1 ]; then
  extArgs+=("--network_args")
  if [ "$loraplus_unet_lr_ratio" -ne 0 ]; then
    extArgs+=(" loraplus_lr_ratio=$loraplus_unet_lr_ratio")
  elif [ "$loraplus_text_encoder_lr_ratio" -eq 0 ]; then
    extArgs+=(" loraplus_lr_ratio=$loraplus_lr_ratio")
  fi
  if [ "$loraplus_text_encoder_lr_ratio" -ne 0 ]; then
    extArgs+=(" loraplus_text_encoder_lr_ratio=$loraplus_text_encoder_lr_ratio")
  fi
fi

if [ $network_dim ]; then
  extArgs+=("--network_dim=$network_dim")
  if [ $network_alpha ]; then
    extArgs+=("--network_alpha=$network_alpha")
  fi
fi

if [ $unet_lr ]; then extArgs+=("--unet_lr=$unet_lr"); fi

if [ $text_encoder_lr ]; then extArgs+=("--text_encoder_lr=$text_encoder_lr"); fi

if [ $prior_loss_weight != "1" ]; then extArgs+=("--prior_loss_weight=$prior_loss_weight"); fi

if [ $keep_tokens -ne 0 ]; then extArgs+=("--keep_tokens=$keep_tokens"); fi

if [ $secondary_separator ]; then extArgs+=("--secondary_separator=$secondary_separator"); fi

if [ $keep_tokens_separator ]; then extArgs+=("--keep_tokens_separator=$keep_tokens_separator"); fi

if [ $enable_wildcard -ne 0 ]; then extArgs+=("--enable_wildcard"); fi

if [ $caption_prefix ]; then extArgs+=("--caption_prefix=$caption_prefix"); fi

if [ $caption_suffix ]; then extArgs+=("--caption_suffix=$caption_suffix"); fi

if [ $alpha_mask ]; then extArgs+=("--alpha_mask"); fi

if [ $training_comment ]; then extArgs+=("--training_comment=$training_comment"); fi

if [ $save_state_on_train_end -ne 0 ]; then 
  extArgs+=("--save_state_on_train_end");
elif [ $save_state -ne 0 ]; then
  extArgs+=("--save_state")
fi

if [ $resume ]; then extArgs+=("--resume=$resume"); fi

if [ $reg_data_dir ]; then extArgs+=("--reg_data_dir=$reg_data_dir"); fi

if [ $train_unet_only -ne 0 ]; then
  extArgs+=("--network_train_unet_only")
elif [ $train_text_encoder_only -ne 0 ]; then
  extArgs+=("--network_train_text_encoder_only")
fi

if [ $network_weights ]; then extArgs+=("--network_weights=$network_weights"); fi

if [ $reg_data_dir ]; then extArgs+=("--reg_data_dir=$reg_data_dir"); fi

if [ $shuffle_caption -ne 0 ]; then extArgs+=("--shuffle_caption"); fi

if [ $persistent_data_loader_workers -ne 0 ]; then extArgs+=("--persistent_data_loader_workers"); fi

if [ $weighted_captions -ne 0 ]; then extArgs+=("--weighted_captions"); fi

if [ $caption_dropout_every_n_epochs != "0" ]; then extArgs+=("--caption_dropout_every_n_epochs=$caption_dropout_every_n_epochs"); fi

if [ $caption_dropout_rate != "0" ]; then extArgs+=("--caption_dropout_rate=$caption_dropout_rate"); fi

if [ $caption_tag_dropout_rate != "0" ]; then extArgs+=("--caption_tag_dropout_rate=$caption_tag_dropout_rate"); fi

if [ $vae ]; then extArgs+=("--vae=$vae"); fi

if [ $cache_latents -ne 0 ]; then
  extArgs+=("--cache_latents")
  if [ $cache_latents_to_disk -ne 0 ]; then
    extArgs+=("--cache_latents_to_disk")
  fi
fi

if [ $mixed_precision ]; then
  extArgs+=("--mixed_precision=$mixed_precision")
fi

if [[ $network_dropout != "0" ]]; then
  enable_lycoris=0
  extArgs+=("--network_dropout=$network_dropout")
  extArgs+=("--scale_weight_norms=$scale_weight_norms")
  if [[ $enable_dylora != "0" && $model != db* ]]; then
    extArgs+=("--network_args rank_dropout=$rank_dropout module_dropout=$module_dropout")
  fi
fi

if [ $enable_lycoris_train == 1 ]; then
  network_module="lycoris.kohya"
  lycoris_network_args=" algo=$algo"
  if [[ $use_scalar -ne 0 ]]; then
    lycoris_network_args+=(" use_scalar=True")
  fi
  if [[ $train_norm -ne 0 ]]; then
    lycoris_network_args+=(" train_norm=True")
  fi
  if [[ $algo != "ia3" ]]; then
    if [[ $algo != "full" ]]; then
      if [[ $conv_dim -ne 0 ]]; then
        lycoris_network_args+=(" conv_dim=$conv_dim conv_alpha=$conv_alpha")
      fi
      if [[ $use_tucker -ne 0 ]]; then
        lycoris_network_args+=(" use_tucker=TFrue")
      fi
    fi
    lycoris_network_args+=(" preset=$preset")
  fi
  if [[ $algo == "locon" && dropout != "0" ]]; then
    lycoris_network_args+=(" dropout=$dropout")
  fi
  if [[ $algo == "lokr" ]]; then
    lycoris_network_args+=(" factor=$factor")
  elif [[ $algo == "dylora" ]]; then
    lycoris_network_args+=(" block_size=$block_size")
  fi
  extArgs+=("--network_args $lycoris_network_args")

elif [ $enable_dylora_train == 1 ]; then
  network_module="networks.dylora"
  extArgs+=("--network_args unit=$unit")
  if [[ $module_dropout != "0" ]]; then
    extArgs+=("module_dropout=$module_dropout")
  fi

elif [ $enable_lora_fa -ne 0 ]; then
  network_module="networks.lora_fa"

elif [ $enable_oft -ne 0 ]; then
  network_module="networks.oft"

elif [ $enable_block_weights == 1 ]; then
  extArgs+=("--network_args down_lr_weight=$down_lr_weight mid_lr_weight=$mid_lr_weight up_lr_weight=$up_lr_weight block_lr_zero_threshold=$block_lr_zero_threshold")
  if [ $enable_block_dim == 1 ]; then
    extArgs+=("block_dims=$block_dims block_alphas=$block_alphas")
    if [ $conv_block_dims ]; then
      extArgs+=("conv_block_dims=$conv_block_dims conv_block_alphas=$conv_block_alphas")
    fi
  fi
fi

if [[ $optimizer_type == "Lion" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 betas=.95,.98")

elif [[ $optimizer_type == "DAdaptation" ]] || [[ $optimizer_type == "DAdaptAdam" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 decouple=True use_bias_correction=True")
  lr="1"
  unet_lr="1"
  text_encoder_lr="1"

elif [[ $optimizer_type == "DAdaptAdan" ]] || [[ $optimizer_type == "DAdaptSGD" ]] || [[ $optimizer_type == "DAdaptAdaGrad" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 betas=.965,.95,.98")
  lr="1"
  unet_lr="1"
  text_encoder_lr="1"

elif [[ $optimizer_type == "adafactor" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args scale_parameter=False warmup_init=False relative_step=False")

elif [[ $optimizer_type == "AdamW" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.1 betas=.9,.95")

elif [[ $optimizer_type == "Prodigy" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 decouple=True use_bias_correction=True d_coef=1.0 d0=$d0 safeguard_warmup=True")
  lr="1"
  unet_lr="1"
  text_encoder_lr="1"

elif [[ $optimizer_type == *AdamW8bit ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 betas=.9,.95 is_paged=True")

elif [[ $optimizer_type == *Lion8bit ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 betas=.95,.98 is_paged=True")

elif [[ $optimizer_type == *ScheduleFree ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.08 weight_lr_power=0")

elif [[ $optimizer_type == "Adan" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01")

elif [[ $optimizer_type == "StableAdamW" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01")

elif [[ $optimizer_type == "Ranger" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01")
fi

if [[ $gradfilter_ema_alpha != "0" ]]; then
  extArgs+=("--gradfilter_ema_alpha=$gradfilter_ema_alpha")
  extArgs+=("--gradfilter_ema_lamb=$gradfilter_ema_lamb")
fi

if [[ $noise_offset != "0" ]]; then
  extArgs+=("--noise_offset=$noise_offset")
  if [[ $adaptive_noise_scale != "0" ]]; then
    extArgs+=("--adaptive_noise_scale=$adaptive_noise_scale")
  fi
  if [[ $noise_offset_random_strength -ne 0 ]]; then
    extArgs+=("--noise_offset_random_strength")
  fi
elif [[ $multires_noise_iterations != "0" ]]; then
  extArgs+=("--multires_noise_iterations=$multires_noise_iterations")
  extArgs+=("--multires_noise_discount=$multires_noise_discount")
fi

if [[ $max_grad_norm != "0" ]]; then extArgs+=("--max_grad_norm=$max_grad_norm"); fi

if [[ $vae_batch_size -ne 0 ]]; then extArgs+=("--vae_batch_size=$vae_batch_size"); fi

if [[ $min_snr_gamma != "0" ]]; then
  extArgs+=("--min_snr_gamma=$min_snr_gamma")
elif [[ $debiased_estimation_loss -ne 0 ]]; then
  extArgs+=("--debiased_estimation_loss")
fi

if [[ $immiscible_noise -ne 0 ]]; then
  extArgs+=("--immiscible_noise")
fi

if [[ $ip_noise_gamma != "0" ]]; then
  extArgs+=("--ip_noise_gamma=$ip_noise_gamma")
  if [[ $ip_noise_gamma_random_strength -ne 0 ]]; then
    extArgs+=("--ip_noise_gamma_random_strength")
  fi
fi

if [[ $gradient_checkpointing -ne 0 ]]; then extArgs+=("--gradient_checkpointing"); fi

if [[ $gradient_accumulation_steps -ne 0 ]]; then extArgs+=("--gradient_accumulation_steps=$gradient_accumulation_steps"); fi

if [ $lr_scheduler ]; then extArgs+=("--lr_scheduler=$lr_scheduler"); fi

if [[ $lr_scheduler_num_cycles -ne 1 ]]; then extArgs+=("--lr_scheduler_num_cycles=$lr_scheduler_num_cycles"); fi

if [[ $lr_scheduler_timescale -ne 0 ]]; then extArgs+=("--lr_scheduler_timescale=$lr_scheduler_timescale"); fi

if [[ $lr_scheduler_min_lr_ratio != "0" ]]; then extArgs+=("--lr_scheduler_min_lr_ratio=$lr_scheduler_min_lr_ratio"); fi

if [[ $lr_warmup_steps -ne 0 ]]; then
  if [[ $gradient_accumulation_steps -ne 0 ]]; then
    $lr_warmup_steps = $lr_warmup_steps * $gradient_accumulation_steps
  fi
  extArgs+=("--lr_warmup_steps=$lr_warmup_steps")
fi

if [[ $lr_decay_steps -ne 0 ]]; then
  if [[ $gradient_accumulation_steps -ne 0 ]]; then
    $lr_decay_steps = $lr_decay_steps * $gradient_accumulation_steps
  fi
  extArgs+=("--lr_decay_steps=$lr_decay_steps")
fi

if [ "$is_v2_model" -eq 1 ]; then
  extArgs+=("--v2")
  min_snr_gamma=0
  debiased_estimation_loss=0
  if [ "$v_parameterization" -eq 1 ]; then
    extArgs+=("--v_parameterization")
    extArgs+=("--scale_v_pred_loss_like_noise_pred")
    extArgs+=("--zero_terminal_snr")
  fi
elif [[ "$model" == hunyuan* ]]; then
  train_script="hunyuan_$train_script"
  extArgs+=("--v_parameterization")
  extArgs+=("--scale_v_pred_loss_like_noise_pred")
  extArgs+=("--zero_terminal_snr")
else
  extArgs+=("--clip_skip=$clip_skip")
fi

if [ $wandb_api_key ]; then
  extArgs+=("--wandb_api_key=$wandb_api_key")
  extArgs+=("--log_with=wandb")
  extArgs+=("--log_tracker_name=$log_tracker_name")
fi

if [ $enable_sample == 1 ]; then
  extArgs+=("--sample_every_n_epochs=$sample_every_n_epochs")
  extArgs+=("--sample_prompts=$sample_prompts")
  extArgs+=("--sample_sampler=$sample_sampler")
fi

if [[ $enable_bucket -ne 0 ]]; then
  extArgs+=("--enable_bucket")
  if [[ $min_bucket_reso != "0" ]]; then
    extArgs+=("--min_bucket_reso=$min_bucket_reso")
  fi
  if [[ $max_bucket_reso != "0" ]]; then
    extArgs+=("--max_bucket_reso=$max_bucket_reso")
  fi
  if [[ $bucket_no_upscale -ne 0 ]]; then
    extArgs+=("--bucket_no_upscale")
  fi
fi

if [[ $torch_compile -ne 0 ]]; then
  extArgs+=("--torch_compile")
  if [ $dynamo_backend ]; then
    extArgs+=("--dynamo_backend=$dynamo_backend")
    launchArgs+=("--dynamo_backend=$dynamo_backend")
  fi
fi

if [[ $base_weights != "" ]]; then
  extArgs+=("--base_weights=$base_weights")
  extArgs+=("--base_weights_multiplier=$base_weights_multiplier")
fi

python -m accelerate.commands.launch ${launchArgs[@]} "./sd-scripts/$train_script.py" \
  --pretrained_model_name_or_path=$pretrained_model \
  --train_data_dir=$train_data_dir \
  --output_dir="./output" \
  --logging_dir="./logs" \
  --resolution=$resolution \
  --max_train_epochs=$max_train_epoches \
  --learning_rate=$lr \
  --output_name=$output_name \
  --train_batch_size=$batch_size \
  --save_every_n_epochs=$save_every_n_epochs \
  --save_precision=$save_precision \
  --seed=$seed \
  --max_token_length=225 \
  --caption_extension=".txt" \
  --save_model_as=$save_model_as \
  ${extArgs[@]}
