'''
 If you use your personal computer, often you will not have the performance needed to work on all this data.
 this script will help you bring out a sample that you can work on.
 if you want to work with all data, ignore this file.
'''
import csv
import shutil

def copyDirectory(src , dest):
	try:
		shutil.copytree(src, dest)
		return 1
	# Directories are the same
	except shutil.Error as e:
		print('Directory not copied. Error: %s' % e)
		return 2
	#Any error saying that the directory doesn't exist
	except OSError as e:
		print('Directory not copied. Error: %s' % e)
		return 3


path_dest = 'C:/Users/chaoj/Desktop/UTMIST/3D-CNN-Gesture-recognition/validation_samples/' # The folder where we will put all our validation samples
path_source = 'C:/Users/chaoj/Desktop/UTMIST/Jester/20bn-jester-v1/' # The folder that contains all the data (Jester 20bn)
csv_file = 'data_csv/jester-v1-validation.csv' # validation csv file

file = open(csv_file, 'r')
reader = csv.reader(file, delimiter=';')
count = 0
simple = 125 # number of samples by class for each class for validation; total: 875 validation samples
#class_ = ['Sliding Two Fingers Up', 'Sliding Two Fingers Down', 'Zooming Out With Two Fingers', 'Zooming In With Two Fingers', 'Sliding Two Fingers Right', 'Sliding Two Fingers Left', 'No gesture']  # The classes we want to use
class_ = ['Sliding Two Fingers Right', 'Sliding Two Fingers Left', 'No gesture']  # The classes we want to use

rows = []

for c in class_:
    for line in reader:
        target = line[1]
        ref = line[0]

        if target == c:
            dep  = copyDirectory(path_source+ref , path_dest+ref)
            if dep == 1:
                count += 1
                rows.append([ref, target])
            if count == simple:
                print("Class: {}; Count: {}".format(c, count))
                count = 0
                break

# you can uncomment this code if you want to create your own csv for the samples
'''with open(path_dest+'validation.csv', 'wt') as f:
	csv_writer = csv.writer(f, quoting = csv.QUOTE_ALL)
	csv_writer.writerows(rows)'''