# Active_Learning_GTG
Active Learning Strategy using Graph Trusduction Algorithm

## Requirements

Change the last row of **requirements.yml** to the path of yout conda enviroments. Then type in the bash the following lines.
```
conda create --name test -f requirements.yml
conda activate test
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
https://github.com/zuliani99/Active_Learning_GTG.git
```

GitHub Token
```
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
```

How to remove the project directory
```
rm -rf Active_Learning_GTG/
```

Connect to VPN UNIVE from Ubuntu
```
nmcli con up id VPN-Ca-Foscari
```

Before run the app on the server, change all the path to the relative path: *app/directory/directory/...*

To download the result plot locally run this commend on a local bash
```
scp rzuliani@157.138.20.71:/home/rzuliani/projects/Active_Learning_GTG/results_1_2_1000.png /home/riccardo/Desktop
```