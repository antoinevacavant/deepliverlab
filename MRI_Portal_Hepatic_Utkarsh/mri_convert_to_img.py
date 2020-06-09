import pylab
import pydicom
import os
from medpy.io import load
image_data, image_header = load('Vein_Sus_Hepatic_Liver.mha')
truth_image_data, truth_image_header = load('Vein_Sus_Hepatic_Truth.mha')
# image_data.shape
# truth_image_data.shape

for i in range(0,64):
  print(i)
  # ImageFile=pydicom.read_file("./3Dircadb1."+str(i)+"/PATIENT_DICOM/image_"+str(j))
  pylab.imsave("Vein_Sus_Hepatic_Liver/"+str(i),image_data[:,:,i],cmap=pylab.cm.bone)
  print(i,"saved image")
  # train_data_gt.append("./3Dircadb1."+str(i)+"/MASKS_DICOM/venoussystem/image_"+str(j))
  # ImageFile=pydicom.read_file("./3Dircadb1."+str(i)+"/MASKS_DICOM/venoussystem/image_"+str(j))
  pylab.imsave("Vein_Sus_Hepatic_Truth/"+str(i),truth_image_data[:,:,i],cmap=pylab.cm.bone)
  print(i,j,"saved label")