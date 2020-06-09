import pylab
import pydicom
import os

train_data = []  # create an empty list
train_data_gt = []  # create an empty list
for i in range(1,21):
  if "venoussystem" in os.listdir("./3Dircadb1."+str(i)+"/MASKS_DICOM"):
    for j in range(0,len(os.listdir("./3Dircadb1."+str(i)+"/MASKS_DICOM/venoussystem"))):
      print(i,j)
      train_data.append("./3Dircadb1."+str(i)+"/PATIENT_DICOM/image_"+str(j))
      ImageFile=pydicom.read_file("./3Dircadb1."+str(i)+"/PATIENT_DICOM/image_"+str(j))
      pylab.imsave("data/train/image/"+str(i)+"_"+str(j),ImageFile.pixel_array,cmap=pylab.cm.bone)
      print(i,j,"saved image")
      train_data_gt.append("./3Dircadb1."+str(i)+"/MASKS_DICOM/venoussystem/image_"+str(j))
      ImageFile=pydicom.read_file("./3Dircadb1."+str(i)+"/MASKS_DICOM/venoussystem/image_"+str(j))
      pylab.imsave("data/train/label/"+str(i)+"_"+str(j),ImageFile.pixel_array,cmap=pylab.cm.bone)
      print(i,j,"saved label")