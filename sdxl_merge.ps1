# LoRA svd_merge script by @bdsqlsz

$save_precision = "fp16" # precision in saving, default float | 保存精度, 可选 float、fp16、bf16, 默认 和源文件相同
$precision = "float" # precision in merging (float is recommended) | 合并时计算精度, 可选 float、fp16、bf16, 推荐float
$sd_model = "./Stable-diffusion/sd_xl_base_1.0_fixvae_fp16_V2.safetensors" # Stable Diffusion model to load: ckpt or safetensors file, merge LoRA models if omitted | dim rank等级, 默认 4
$models = "D:\sd-webui-aki-v4.1\models\Lora\animescreenshot_xl_last.safetensors" # original LoRA model path need to resize, save as cpkt or safetensors | 需要合并的模型路径, 保存格式 cpkt 或 safetensors，多个用空格隔开
$save_to = "./Stable-diffusion/sd_xl_anime_1.0.safetensors" # output LoRA model path, save as ckpt or safetensors | 输出路径, 保存格式 cpkt 或 safetensors
$ratios = "1.0" # ratios for each model / LoRA模型合并比例，数量等于模型数量，多个用空格隔开

# Activate python venv
.\venv\Scripts\activate

$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$ext_args = [System.Collections.ArrayList]::new()

[void]$ext_args.Add("--models")
foreach ($model in $models.Split(" ")) {
    [void]$ext_args.Add($model)
}

[void]$ext_args.Add("--ratios")
foreach ($ratio in $ratios.Split(" ")) {
    [void]$ext_args.Add([float]$ratio)
}

# run svd_merge
accelerate launch --num_cpu_threads_per_process=8 "./sd-scripts/networks/sdxl_merge_lora.py" `
	--save_precision=$save_precision `
	--precision=$precision `
	--sd_model=$sd_model `
	--save_to=$save_to `
	$ext_args 

Write-Output "SVD Merge finished"
Read-Host | Out-Null ;
