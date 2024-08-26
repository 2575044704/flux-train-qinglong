#!/bin/bash
# tagger script by @bdsqlsz
# Train data path
train_data_dir="./input" # input images path | 图片输入路径
repo_id="SmilingWolf/wd-swinv2-tagger-v3" # model repo id from huggingface |huggingface模型repoID
model_dir="" # model dir path | 本地模型文件夹路径
batch_size=12 # batch size in inference 批处理大小，越大越快
max_data_loader_n_workers=0 # enable image reading by DataLoader with this number of workers (faster) | 0最快
thresh=0.27 # concept thresh | 最小识别阈值
general_threshold=0.27 # general threshold | 总体识别阈值 
character_threshold=0.1 # character threshold | 人物姓名识别阈值 
recursive=0 # search for images in subfolders recursively | 递归搜索下层文件夹，1为开，0为关
frequency_tags=0 # order by frequency tags | 从大到小按识别率排序标签，1为开，0为关
onnx=1 #使用ONNX格式，V3必选

#Tag Edit | 标签编辑
remove_underscore=1 # remove_underscore | 下划线转空格，1为开，0为关 
undesired_tags="" # no need tags | 排除标签
use_rating_tags=0 #使用评分标签
use_rating_tags_as_last_tag=0 #分类标签放最后
character_tags_first=1 #角色标签放在前面
character_tag_expand=1 #人物 系列拆分，chara_name_(series) 变为 chara_name, series.
always_first_tags="" #指定标签放最前，当图像中出现某个标签时，总是先输出该标签。可以指定多个标签，以逗号分隔
tag_replacement="" #执行标记替换。指定格式为 tag1,tag2;tag3,tag4。如果使用 , 和 ;，请用\转义。例如，指定 aira tsubase,aira tsubase（uniform）（当您要训练特定服装时）、aira tsubase,aira tsubase\, heir of shadows（当标签中不包括系列名称时）。
remove_parents_tag=1 #去除父标签

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================

export HF_HOME="huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export TF_CPP_MIN_LOG_LEVEL=3
extArgs=()

if [ -n "$repo_id" ]; then
  extArgs+=( "--repo_id=$repo_id" )
fi

if [ -n "$model_dir" ]; then
  extArgs+=( "--model_dir=$model_dir" )
fi

if [[ $batch_size -ne 0 ]]; then
  extArgs+=( "--batch_size=$batch_size" )
fi

if [ -n "$max_data_loader_n_workers" ]; then
  extArgs+=( "--max_data_loader_n_workers=$max_data_loader_n_workers" )
fi

if [ -n "$general_threshold" ]; then
  extArgs+=( "--general_threshold=$general_threshold" )
fi

if [ -n "$character_threshold" ]; then
  extArgs+=( "--character_threshold=$character_threshold" )
fi

if [ "$remove_underscore" -eq 1 ]; then
  extArgs+=( "--remove_underscore" )
fi

if [ -n "$undesired_tags" ]; then
  extArgs+=( "--undesired_tags=$undesired_tags" )
fi

if [ "$recursive" -eq 1 ]; then
  extArgs+=( "--recursive" )
fi

if [ "$frequency_tags" -eq 1 ]; then
  extArgs+=( "--frequency_tags" )
fi

if [ "$onnx" -eq 1 ]; then
  extArgs+=( "--onnx" )
fi

if [ "$character_tags_first" -eq 1 ]; then
  extArgs+=( "--character_tags_first" )
fi

if [ "$character_tag_expand" -eq 1 ]; then
  extArgs+=( "--character_tag_expand" )
fi

if [ "$use_rating_tags" -eq 1 ]; then
  extArgs+=( "--use_rating_tags" )
  if [ "$use_rating_tags_as_last_tag" -eq 1 ]; then
    extArgs+=( "--use_rating_tags_as_last_tag" )
  fi
fi

if [ -n "$always_first_tags" ]; then
  extArgs+=( "--always_first_tags=$always_first_tags" )
fi

if [ -n "$tag_replacement" ]; then
  extArgs+=( "--tag_replacement=$tag_replacement" )
fi

if [ -n "$remove_parents_tag" ]; then
  extArgs+=( "--remove_parents_tag" )
fi

# run tagger
accelerate launch --num_cpu_threads_per_process=8 "./sd-scripts/finetune/tag_images_by_wd14_tagger.py" \
  $train_data_dir \
  --thresh=$thresh \
  --caption_extension .txt \
  ${extArgs[@]}
