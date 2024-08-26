# tagger script by @bdsqlsz

# Train data path
$train_data_dir = "./input/test" # input images path | 图片输入路径
$repo_id = "SmilingWolf/wd-swinv2-tagger-v3" # model repo id from huggingface |huggingface模型repoID
$model_dir = "wd14_tagger_model" # model dir path | 本地模型文件夹路径
$batch_size = 12 # batch size in inference 批处理大小，越大越快
$max_data_loader_n_workers = 0 # enable image reading by DataLoader with this number of workers (faster) | 0最快
$thresh = 0.27 # concept thresh | 最小识别阈值
$general_threshold = 0.27 # general threshold | 总体识别阈值 
$character_threshold = 0.1 # character threshold | 人物姓名识别阈值
$recursive = 1 # search for images in subfolders recursively | 递归搜索下层文件夹，1为开，0为关
$frequency_tags = 0 # order by frequency tags | 从大到小按识别率排序标签，1为开，0为关
$onnx = 1 #使用ONNX模型

#Tag Edit | 标签编辑
$remove_underscore = 1 # remove_underscore | 下划线转空格，1为开，0为关 
$undesired_tags = "simple background,white background," # no need tags | 排除标签
$use_rating_tags = 0 #使用评分标签
$use_rating_tags_as_last_tag= 0 #分类标签放最后
$character_tags_first = 1 #角色标签放在前面
$character_tag_expand = 1 #人物 系列拆分，chara_name_(series) 变为 chara_name, series.
$always_first_tags = "" #指定标签放最前，当图像中出现某个标签时，总是先输出该标签。可以指定多个标签，以逗号分隔
$tag_replacement = "" #执行标记替换。指定格式为 tag1,tag2;tag3,tag4。如果使用 , 和 ;，请用\转义。例如，指定 aira tsubase,aira tsubase（uniform）（当您要训练特定服装时）、aira tsubase,aira tsubase\, heir of shadows（当标签中不包括系列名称时）。
$remove_parents_tag = 1

# Activate python venv
.\venv\Scripts\activate

$Env:HF_HOME = "huggingface"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$ext_args = [System.Collections.ArrayList]::new()

if ($repo_id) {
  [void]$ext_args.Add("--repo_id=$repo_id")
}

if ($model_dir) {
  [void]$ext_args.Add("--model_dir=$model_dir")
}

if ($batch_size) {
  [void]$ext_args.Add("--batch_size=$batch_size")
}

if ($max_data_loader_n_workers) {
  [void]$ext_args.Add("--max_data_loader_n_workers=$max_data_loader_n_workers")
}

if ($general_threshold) {
  [void]$ext_args.Add("--general_threshold=$general_threshold")
}

if ($character_threshold) {
  [void]$ext_args.Add("--character_threshold=$character_threshold")
}

if ($remove_underscore) {
  [void]$ext_args.Add("--remove_underscore")
}

if ($undesired_tags) {
  [void]$ext_args.Add("--undesired_tags=$undesired_tags")
}

if ($recursive) {
  [void]$ext_args.Add("--recursive")
}

if ($frequency_tags) {
  [void]$ext_args.Add("--frequency_tags")
}

if ($onnx) {
  [void]$ext_args.Add("--onnx")
}

if ($character_tags_first) {
  [void]$ext_args.Add("--character_tags_first")
}

if ($character_tag_expand) {
  [void]$ext_args.Add("--character_tag_expand")
}

if ($use_rating_tags) {
  [void]$ext_args.Add("--use_rating_tags")
  if ($use_rating_tags_as_last_tag) {
    [void]$ext_args.Add("--use_rating_tags_as_last_tag")
  }
}

if ($always_first_tags) {
  [void]$ext_args.Add("--always_first_tags=$always_first_tags")
}

if ($tag_replacement) {
  [void]$ext_args.Add("--tag_replacement=$tag_replacement")
}

if ($remove_parents_tag) {
  [void]$ext_args.Add("--remove_parents_tag")
}

# run tagger
accelerate launch --num_cpu_threads_per_process=8 "./sd-scripts/finetune/tag_images_by_wd14_tagger.py" `
  $train_data_dir `
  --thresh=$thresh `
  --caption_extension .txt `
  $ext_args

Write-Output "Tagger finished"
Read-Host | Out-Null ;
