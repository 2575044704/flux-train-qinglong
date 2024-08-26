#!/usr/bin/bash
#export PIP_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple"
#export HF_ENDPOINT="https://hf-mirror.com"

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
create_venv=true

while [ -n "$1" ]; do
    case "$1" in
        --disable-venv)
            create_venv=false
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if $create_venv; then
    echo "Creating python venv..."
    python3 -m venv venv
    source "$script_dir/venv/bin/activate"
    echo "active venv"
fi

echo "Installing torch & xformers..."

cuda_version=$(nvcc --version | grep 'release' | sed -n -e 's/^.*release \([0-9]\+\.[0-9]\+\),.*$/\1/p')
cuda_major_version=$(echo "$cuda_version" | awk -F'.' '{print $1}')
cuda_minor_version=$(echo "$cuda_version" | awk -F'.' '{print $2}')

echo "Cuda Version:$cuda_version"

echo "install torch 2.3.1+cu121"
pip install xformers==0.0.27post2 torchvision==0.19.0
echo "Installing deps..."
cd "$script_dir/sd-scripts" || exit

pip install --upgrade -r requirements.txt
pip install --upgrade lycoris-lora dadaptation fastapi uvicorn wandb
pip install --upgrade --no-deps pytorch-optimizer
pip install --upgrade schedulefree

cd "$script_dir" || exit

echo "Install completed"
