import os 

def print_annot(filenum, pednum, bboxlist, labelfile):
	imagename = str(filenum) + '.jpg'
	imagepath = os.path.join(videopath,imagename)
	#print(videopath)
	print(imagepath, file=labelfile)
	print(pednum, file=labelfile)
	for bbox in bboxlist:
		boxstr = ' '.join(bbox)
		print(boxstr, file=labelfile)
	'''
	labelfile.write(imagepath + '\n')
	labelfile.write(str(pednum) + '\n')
	for bbox in bboxlist:
		boxstr = ' '.join(bbox)
		labelfile.write(boxstr + '\n')
	'''
	return
def parse_annot(videopath, labelfile):
	#filepath = 'cal_ped/set01/V000.txt'
	#imagedir = 'cal_ped/set01/V000'
	filepath = videopath + '.txt'
	f = open(filepath)	
	filenum = 1
	pednum = 0
	bboxlist = []
	for line in f:
		data = line.split()
		#data = f.readline().rstrip().split()
		#print(filenum == int(data[0]))
		if filenum == int(data[0]):
			pednum += 1
		else:
			print_annot(filenum, pednum, bboxlist, labelfile)
			filenum = int(data[0])
			pednum = 1
			bboxlist = []
		bboxlist.append(data[1:5])
		#print(filenum, int(data[0]), pednum)	
	print_annot(filenum, pednum, bboxlist, labelfile)
	f.close()
	return


root_dir = 'cal_ped'
setnames = os.listdir(root_dir)
for setname in setnames:
	setpath = os.path.join(root_dir, setname)
	#print(setpath)
	if os.path.isdir(setpath):
		labelpath = setname + '_annot.txt'
		print(labelpath)	
		labelfile = open(labelpath,'w')	
	if not setname.startswith('.') and os.path.isdir(setpath):
		videonames = os.listdir(setpath)
		for videoname in videonames:
			videopath = os.path.join(setpath, videoname)
			if not videoname.startswith('.') and os.path.isdir(videopath):
				#print(videopath)
				#print(os.path.isfile(videopath + '.txt'))
				parse_annot(videopath, labelfile)
	#labelfile.close()