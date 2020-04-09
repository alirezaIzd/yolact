import json
import os, errno
from xml.dom import minidom
from bs4 import BeautifulSoup
import json
import base64
import numpy as np
import cv2
from PIL import Image
import random
import argparse
import os.path as osp
import sys
import io
import PIL.Image
from augmentor import Augmentor
from config import config
import csv
import shutil
import numpy as np



def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return image

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in PIL.ExifTags.TAGS
    }

    orientation = exif.get('Orientation', None)

    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image

def convert_2D_to_3D(img):
    
    np_img = np.array(img)
    backtorgb = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    img_3D = Image.fromarray(backtorgb)

    return img_3D



def load_image_file(filename):
    try:
        image_pil = PIL.Image.open(filename)
    except IOError:
        print('Failed opening image file: {}'.format(filename))
        return

    # apply orientation to image according to exif
    image_pil = apply_exif_orientation(image_pil)

    with io.BytesIO() as f:
        ext = osp.splitext(filename)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            format = 'JPEG'
        else:
            format = 'PNG'
        image_pil.save(f, format=format)
        f.seek(0)
        return f.read()



def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


def noise_creator(image, label, ag):
    
    # add noise to images and corresponding label
    if ag is not None:
        image, label = ag.addNoise(image, label)
        return image, label

def getText(nodelist):
    # Iterate all Nodes aggregate TEXT_NODE
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
        else:
            # Recursive
            rc.append(getText(node.childNodes))
    return ''.join(rc)





def create_json(idx, new_path, annotaded_dir, kind_of_use, label):

    

        
    xcenter, ycenter = label[0], label[1]
    width, height = label[2], label[3]
    angle = label[4]

    theta = np.deg2rad(np.arange(0.0, 360.0, 1))

    x = 0.5 * width * np.cos(theta)
    y = 0.5 * height * np.sin(theta)

    rtheta = np.radians(angle)

    R = np.array([
    [np.cos(rtheta), -np.sin(rtheta)],
    [np.sin(rtheta),  np.cos(rtheta)],
    ])

    x, y = np.dot(R, np.array([x, y]))
    x += xcenter
    y += ycenter
    
    points = []
    for i in range(0, len(x)):
    
        if x[i]<0 :
        
            x[i] = 0
            
        if x[i]> 192 :
        
            x[i] = 192
            
        if y[i]<0 :
        
            y[i] = 0
            
        if y[i]> 192 :
        
            y[i] = 192    
            
        points.append([x[i],y[i]])
    
    img = load_image_file(new_path)
    data = {
            "version":"3.21.1",
            "flags":{},
            "shapes": [
                {
                "label": "pupil",
                "line_color": None,
                "fill_color": None,
                "points":points,
            "shape_type": "polygon",
            "flags": {}
            }
          ],
          "lineColor": [
            0,
            255,
            0,
            128
          ],
          "fillColor": [
            255,
            0,
            0,
            128
          ],
        "imagePath": str(idx) +".jpg",
        "imageData": base64.encodebytes(img).decode("utf-8"),
        "imageHeight": 192,
        "imageWidth": 192
        }
    json.dumps(data, indent=4)
    with open(annotaded_dir + "/" + kind_of_use + "/"+ str(idx) + ".json",'w') as json_file:
        json.dump(data, json_file)
       
       


def data_set_changer (all_image_valid_list, valid_list_xml, valid_list, annotaded_dir, kind_of_use, ag_flag, ag_percentage, eval_dataset_flag):
    
    ag_list= []
   
    
    
    
    if ag_flag=="True" and int(ag_percentage)>0 and  eval_dataset_flag == "False":
    	
        ag = Augmentor('./data/pupil_noisy_videos', config)
        ag_list = random.sample(range(0,len(valid_list)),len(valid_list) * int(ag_percentage)//100)
        
    count =  0 
    csv_columns = ['ID', 'True_Elipse_X', 'True_Elipse_Y', 'True_Elipse_W', 'True_Elipse_H', 'True_Elipse_Alpha', 'Predict_Elipse_X', 'Predict_Elipse_Y','Predict_Elipse_W', 'Predict_Elipse_H', 'Predict_Elipse_Alpha',  'True_Box_X1', 'True_Box_Y1', 'True_Box_X2', 'True_Box_Y2', 'True_Box_Center_X', 'True_Box_Center_Y','Predict_Box_X1', 'Predict_Box_Y1', 'Predict_Box_X2', 'Predict_Box_Y2', 'Predict_Box_Center_X', 'Predict_Box_Center_Y']
    pupil_center =[]
    
    for idx, current_index in enumerate(valid_list): 
        
        img = Image.open(all_image_valid_list[current_index])
        img_3D = convert_2D_to_3D(img)
        new_path = annotaded_dir + "/" + kind_of_use + "/" + str(count) + ".jpg"
        img_3D.save(new_path)
        
        xml_dir = valid_list_xml[current_index]
        xmldoc = minidom.parse(xml_dir)
        
        in_label = []
        in_label.append(getText(xmldoc.getElementsByTagName("x")[0].childNodes))
        in_label.append(getText(xmldoc.getElementsByTagName("y")[0].childNodes))
        in_label.append(getText(xmldoc.getElementsByTagName("w")[0].childNodes))
        in_label.append(getText(xmldoc.getElementsByTagName("h")[0].childNodes))
        in_label.append(getText(xmldoc.getElementsByTagName("a")[0].childNodes))
        label = np.asarray(in_label, dtype=np.float32)
        
        
        if label[0] <= 0 or  label[0] >= 192:
            print("label for {0} is out of bound".format(img))
            continue

        if  label[1] <= 0 or  label[1] >= 192:
            print("label for {0} is out of bound".format(img))
            continue
            
        pupil_center.append({'ID':count, 'True_Elipse_X':label[0], 'True_Elipse_Y':label[1], 'True_Elipse_W':label[2], 'True_Elipse_H':label[3], 'True_Elipse_Alpha':label[4], 'Predict_Elipse_X':"",'Predict_Elipse_Y':"", 'Predict_Elipse_W':"", 'Predict_Elipse_H':"", 'Predict_Elipse_Alpha':"", 'True_Box_X1':"", 'True_Box_Y1':"", 'True_Box_X2':"", 'True_Box_Y2':"", 'True_Box_Center_X':"", 'True_Box_Center_Y':"", 'Predict_Box_X1':"", 'Predict_Box_Y1':"", 'Predict_Box_X2':"", 'Predict_Box_Y2':"", 'Predict_Box_Center_X':"", 'Predict_Box_Center_Y':""})
        create_json(count, new_path, annotaded_dir, kind_of_use, label)
        count = count + 1
        
        if len(ag_list) > 0 and idx in ag_list:
         
            new_path = annotaded_dir + "/" + kind_of_use + "/" + str(count) + ".jpg"
            image, noise_label = noise_creator(img, label, ag)
            image, noise_label = noise_creator(img, label, ag)
            noise_label = np.asarray(noise_label, dtype=np.float32)
            pupil_center.append({'ID':count, 'True_Elipse_X':noise_label[0], 'True_Elipse_Y':noise_label[1], 'True_Elipse_W':noise_label[2], 'True_Elipse_H':noise_label[3], 'True_Elipse_Alpha':noise_label[4],'Predict_Elipse_X':"", 'Predict_Elipse_Y':"",  'Predict_Elipse_W':"", 'Predict_Elipse_H':"", 'Predict_Elipse_Alpha':"",'True_Box_X1':"", 'True_Box_Y1':"", 'True_Box_X2':"", 'True_Box_Y2':"", 'True_Box_Center_X':"", 'True_Box_Center_Y':"", 'Predict_Box_X1':"", 'Predict_Box_Y1':"", 'Predict_Box_X2':"", 'Predict_Box_Y2':"", 'Predict_Box_Center_X':"", 'Predict_Box_Center_Y':""})
            img_3D = convert_2D_to_3D(image)
            img_3D.save(new_path)
            create_json(str(count), new_path, annotaded_dir, kind_of_use, noise_label)
            count = count + 1
    
    
    
    
    if kind_of_use != "train":
        csv_file = "./results/pupil_Detection_result/data_info.csv"
        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in pupil_center:
                    writer.writerow(data)
        except IOError:
            print("I/O error")
	


            
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--images_dir', help='Input Image Subdirectory')
    parser.add_argument('--annotaded_dir', help='Output Annotaded Directory')
    parser.add_argument('--ag_flag', help='Agumentor Flag')
    parser.add_argument('--ag_percentage', default=0, help='Percentage of Agumentor')
    parser.add_argument('--eval_dataset_flag', default=False, help='Create Just Evaluate Dataset')
    args = parser.parse_args()
    

  
    shutil.rmtree(args.annotaded_dir, ignore_errors=True)
        
    os.makedirs(args.annotaded_dir)
    
    
    all_image_valid_list = []
    valid_list_xml = []
    list_of_files = getListOfFiles(args.images_dir)

    for idx, current_file in enumerate(list_of_files):
    
        if ".jpg" in current_file:
            xml_dir = "/".join(current_file.split("/")[0:-1])+"/"+current_file.split("/")[-1].replace("in.jpg", "gt.xml")
            if xml_dir in list_of_files:
                all_image_valid_list.append(current_file) 
                valid_list_xml.append(xml_dir)
    
    
    result_export_path = "./results/pupil_Detection_result"
    shutil.rmtree( result_export_path, ignore_errors=True)
    os.makedirs(result_export_path)
        
    
    if args.eval_dataset_flag=="False":
    
        temp_list = random.sample(range(0, len(all_image_valid_list)), len(all_image_valid_list))
        train_list = temp_list[0:((len(temp_list)*80)//100)]
        test_list = temp_list[len(train_list):len(temp_list)]
        
        os.makedirs(args.annotaded_dir + "/train")
        os.makedirs(args.annotaded_dir + "/test")
       
        data_set_changer(all_image_valid_list, valid_list_xml, train_list, args.annotaded_dir, "train", args.ag_flag, args.ag_percentage, args.eval_dataset_flag)
        data_set_changer(all_image_valid_list, valid_list_xml, test_list, args.annotaded_dir, "test", args.ag_flag, args.ag_percentage, args.eval_dataset_flag)
    
    if args.eval_dataset_flag == "True":
        
        os.makedirs(args.annotaded_dir + "/test")
        eval_list = random.sample(range(0, len(all_image_valid_list)), len(all_image_valid_list))      
        data_set_changer(all_image_valid_list, valid_list_xml, eval_list, args.annotaded_dir, "test", args.ag_flag, args.ag_percentage, args.eval_dataset_flag)
        
    
    
  
    print('Creating annotaded directory:', args.annotaded_dir)
    
    
	

if __name__ == '__main__':
    main()

