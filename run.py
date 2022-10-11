import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='name', type=str, help='yolov5 epoch')
args = parser.parse_args()
print(args.name)

file = open("/workspace16/Shayan/yolov5_edited/yolov5/{}.txt".format(args.name)) 
list = file.readlines()    
for item in list:
    item = item.replace("\n","")
    hold = item.split(",")
    try:
        os.system("python -u train.py --data test.yaml --weights '' --cfg yolov5x.yaml --batch-size 8 --name [{}] --epoch {} --min_radius {} --max_radius {} --min_rotation {} --max_rotation {} --min_cageScaling {} --max_cageScaling {} --cage {}".format(item,hold[7].split("=")[1],hold[0].split("=")[1],hold[1].split("=")[1],hold[2].split("=")[1],hold[3].split("=")[1],hold[4].split("=")[1],hold[5].split("=")[1],hold[6].split("=")[1]))
    except Exception as e:
        print(e)  

