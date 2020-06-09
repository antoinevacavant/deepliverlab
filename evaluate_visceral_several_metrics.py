# ./EvaluateSegmentation /home/utkarsh/Documents/Pascal/testing_ground_truths/1_2.png /home/utkarsh/Documents/Pascal/unet_200_epochs_testing_results/final_unet_pred_1_2.png -use DICE | awk ' /'DICE'/ {print $3} '
import subprocess
import os
# output = subprocess.check_output("./EvaluateSegmentation /home/utkarsh/Documents/Pascal/testing_ground_truths/1_2.png /home/utkarsh/Documents/Pascal/unet_200_epochs_testing_results/final_unet_pred_1_2.png -use DICE | awk ' /'DICE'/ {print $3} '", shell=True)
# print(float(output)+1)
# print("should be 1.924264")

# print("****start all*****")
truths_file_list = sorted(os.listdir("/home/utkarsh/Documents/Pascal/testing_ground_truths/"))
preds_file_list = sorted(os.listdir("/home/utkarsh/Documents/Pascal/unet_200_epochs_testing_results/"))

# print(len(file_list), "files found")

metrics = ["DICE", "JACRD", "AUC", "KAPPA", "RNDIND", "ADJRIND", "ICCORR", "VOLSMTY", "MUTINF", "HDRFDST", "AVGDIST", "MAHLNBS", "VARINFO", "GCOERR", "PROBDST", "SNSVTY", "SPCFTY", "PRCISON", "FMEASR", "ACURCY", "FALLOUT", "TP", "FP", "TN", "FN", "REFVOL", "SEGVOL"]



for metric in metrics:
	sum = 0.0
	count = 0.0

	for i,file in enumerate(preds_file_list):
		# print(file)
		# print(truths_file_list[i])
		output = subprocess.check_output("./EvaluateSegmentation /home/utkarsh/Documents/Pascal/testing_ground_truths/"+truths_file_list[i]+" /home/utkarsh/Documents/Pascal/unet_200_epochs_testing_results/"+file+" -use "+metric+" | awk ' /'"+metric+"'/ {print $3} '", shell=True)
		# print(output)
		if output == b'':
			continue
		sum = sum + float(output)
		count = count+1.0

	print(metric,":", sum/count)
# print(count)