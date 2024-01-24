# Active_Learning_GTG
Active Learning Strategy using Graph Trusduction Algorithm

## Requirements

Change the last row of **requirements.txt** to the path of yout conda enviroments. Then type in the bash the following lines.
```
conda create --name <env> --file requirements.txt
conda activate <env>
```

## Applciation Start Up
```
ssh rzuliani@157.138.20.71
cd app
python main.py 
```

## Notes
Github url repo:
```
git clone https://github.com/zuliani99/Active_Learning_GTG.git
```

GitHub Token
```
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
```

Connect to VPN UNIVE from Ubuntu
```
nmcli con up id VPN-Ca-Foscari
```

How to check the usage of GPU memory
```
nvidia-smi --query-gpu=gpu_name,memory.used,memory.free,memory.total --format=csv
```
