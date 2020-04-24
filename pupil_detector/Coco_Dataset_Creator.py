import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import uuid
import PIL
import numpy as np
from PIL import Image, ImageDraw
from util  import shape_to_mask
import sys
from tempfile import NamedTemporaryFile
import shutil
import csv


try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)



def update_csv_result(image_index, bbox):

    filename = './results/pupil_Detection_result/data_info.csv'
    
    tempfile = NamedTemporaryFile(mode='w', delete=False)               

    fields = ['ID', 'Original_Ellipse_Center_X', 'Original_Ellipse_Center_Y','Original_Ellipse_W','Original_Ellipse_H','Original_Ellipse_Alpha', 'Predict_Ellipse_Center_X', 'Predict_Ellipse_Center_Y', 'Predict_Ellipse_W','Predict_Ellipse_H','Predict_Ellipse_Alpha',  'Original_Box_X1', 'Original_Box_Y1', 'Original_Box_X2', 'Original_Box_Y2','Predict_Box_X1', 'Predict_Box_Y1', 'Predict_Box_X2', 'Predict_Box_Y2','Ellipse_Center_Mean_Squared_Error']
    
   
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]


    with open(filename, 'r') as csvfile, tempfile:
        reader = csv.DictReader(csvfile, fieldnames=fields)
        writer = csv.DictWriter(tempfile, fieldnames=fields)
        for row in reader:
        
            if row['ID'] == str(image_index):
   
                row['Original_Box_X1'] = x1
                row['Original_Box_Y1'] = y1
                row['Original_Box_X2'] = x2
                row['Original_Box_Y2'] = y2

                
                
                row['Predict_Ellipse_Center_X'] =  row['Predict_Ellipse_Center_X'] 
                row['Predict_Ellipse_Center_Y'] = row['Predict_Ellipse_Center_Y']
                row['Predict_Ellipse_W'] = row['Predict_Ellipse_W']
                row['Predict_Ellipse_H'] = row['Predict_Ellipse_H']
                row['Predict_Ellipse_Alpha'] = row['Predict_Ellipse_Alpha']
                
                row['Original_Ellipse_Center_X'] = row['Original_Ellipse_Center_X'] 
                row['Original_Ellipse_Center_Y'] =  row['Original_Ellipse_Center_Y'] 
                row['Original_Ellipse_W'] =  row['Original_Ellipse_W'] 
                row['Original_Ellipse_H'] =  row['Original_Ellipse_H'] 
                row['Original_Ellipse_Alpha'] =  row['Original_Ellipse_Alpha'] 
                row['Predict_Box_X1'] = row['Predict_Box_X1'] 
                row['Predict_Box_Y1'] = row['Predict_Box_Y1'] 
                row['Predict_Box_X2'] = row['Predict_Box_X2']
                row['Predict_Box_Y2'] = row['Predict_Box_Y2']
                row['Ellipse_Center_Mean_Squared_Error'] = row['Ellipse_Center_Mean_Squared_Error']
           

            row = {'ID': row['ID'], 'Original_Ellipse_Center_X':row['Original_Ellipse_Center_X'],'Original_Ellipse_Center_Y':row['Original_Ellipse_Center_Y'], 'Original_Ellipse_W': row['Original_Ellipse_W'] ,'Original_Ellipse_H': row['Original_Ellipse_H'] , 'Original_Ellipse_Alpha': row['Original_Ellipse_Alpha'] ,'Predict_Ellipse_Center_X': row['Predict_Ellipse_Center_X'],  'Predict_Ellipse_Center_Y': row['Predict_Ellipse_Center_Y'], 'Predict_Ellipse_W': row['Predict_Ellipse_W'], 'Predict_Ellipse_H': row['Predict_Ellipse_H'], 'Predict_Ellipse_Alpha': row['Predict_Ellipse_Alpha'], 'Original_Box_X1':  row['Original_Box_X1'], 'Original_Box_Y1': row['Original_Box_Y1'], 'Original_Box_X2' : row['Original_Box_X2'], 'Original_Box_Y2' : row['Original_Box_Y2'], 'Predict_Box_X1': row['Predict_Box_X1'], 'Predict_Box_Y1': row['Predict_Box_Y1'] , 'Predict_Box_X2': row['Predict_Box_X2'], 'Predict_Box_Y2':  row['Predict_Box_Y2'],'Ellipse_Center_Mean_Squared_Error':  row['Ellipse_Center_Mean_Squared_Error'] }
            writer.writerow(row)

    shutil.move(tempfile.name, filename)
    

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', help='input annotated directory')
    parser.add_argument('--output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    args = parser.parse_args()



    shutil.rmtree(args.output_dir, ignore_errors=True)

    os.makedirs(args.output_dir)
    #os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
    print('Creating Json dataset:', args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(
            supercategory=None,
            id=class_id,
            name=class_name,
        ))

    out_ann_file = osp.join(args.output_dir, 'annotations.json')
    label_files = glob.glob(osp.join(args.input_dir, '*.json'))
    
    
    for image_id, label_file in enumerate(label_files):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            label_data = json.load(f)

        base = osp.splitext(osp.basename(label_file))[0]
        out_img_file = base + '.jpg'
   

        img_file = args.input_dir+"/"+out_img_file
        img = np.asarray(PIL.Image.open(img_file).convert('RGB'))
        data['images'].append(dict(
            license=0,
            url=None,
            file_name= out_img_file,
            height=img.shape[0],
            width=img.shape[1],
            date_captured=None,
            id=label_file.split("/")[-1].split(".")[0],
        ))

        masks = {}                                     # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_data['shapes']:
            points = shape['points']
            label = shape['label']
            group_id = shape.get('group_id')
            shape_type = shape.get('shape_type')
            mask = shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            points = np.asarray(points).flatten().tolist()
            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()
            
            image_index = label_file.split("/")[-1].split(".")[0]
            
            if "test" in args.output_dir:
                update_csv_result(image_index, bbox)

            data['annotations'].append(dict(
                #id=len(data['annotations']),
                id = image_index,
                image_id = image_index,
                category_id=cls_id,
                segmentation=segmentations[instance],
                area=area,
                bbox=bbox,
                iscrowd=0,
            ))

    with open(out_ann_file, 'w') as f:
        json.dump(data, f)



if __name__ == '__main__':
    main()
        
