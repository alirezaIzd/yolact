1- Using Anaconda:

		conda env create -f environment.yml

 
2- conda activate yolact-env



3- please check openncv and Install it if you don't have:
		

		pip install opencv-python

4- Install bs4:
		 
	
		conda install -c anaconda beautifulsoup4


5- Install pycocotools: 

				
		pip install pycocotools


6- Put your data at directory:


		./Original-data


7- To create json file with coco Stndard with image and xml file of pupils' postion do just "Eavluate data":
	
		python ./pupil_detector/export_coco_standard.py --images_dir=./data/pupil_original-data/ --annotaded_dir=./data/pupil_annotaded_data --ag_flag=True  --ag_percentage=100  --eval_dataset_flag=True


8- To create Noise for your data please add some Noisy video at directory:
	
		./noisy_videos


9- To create json file with coco Stndard with image and xml file of pupils' postion do for "Train/test":


		python ./pupil_detector/export_coco_standard.py --images_dir=./data/pupil_original-data/ --annotaded_dir=./data/pupil_annotaded_data --ag_flag=True  --ag_percentage=100  --eval_dataset_flag=False

10- To create Coco Dataset for trainand test and evaluate the model do:
			

		python ./pupil_detector/Coco_Dataset_Creator.py  --input_dir=./data/pupil_annotaded_data/train --output_dir=./data/pupil_annotaded_data/yolact_datasets_train --labels=./pupil_detector/labels.txt
		python ./pupil_detector/Coco_Dataset_Creator.py  --input_dir=./data/pupil_annotaded_data/test --output_dir=./data/pupil_annotaded_data/yolact_datasets_test --labels=./pupil_detector/labels.txt
		


11- Downlaod the weights from:

	 	
		https://github.com/dbolya/yolact 


12- Create weights director and copy it at weights:

		
		./yolact/weights 



12- For train and evaluate your model please check the main git:

	
		from https://github.com/dbolya/yolact


13- For tune the model please check:


		https://github.com/dbolya/yolact/issues/334
		https://github.com/dbolya/yolact/issues/36
		https://github.com/dbolya/yolact/issues/206




14- For train the moedl RUN:

		python train.py --config=yolact_base_config --resume=./weights/yolact_plus_resnet50_54_800000.pth --batch_size=5 --start_iter=0




15- For Evaluate your Dataset run:
			
		
		python eval.py --trained_model=/home/izad/yolact/weights/yolact_base_3199_12800.pth --score_threshold=0.15 --top_k=15 --display --save_pupil_evaluate_image=True

	
		
16- Please check the Result directory:
	
	
		./pupil_Detection_result. 

		







		
	
		




