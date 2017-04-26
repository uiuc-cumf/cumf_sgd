#! /usr/bin/python

import os,sys

if len(sys.argv) != 3:
	print "usage: ./to_txt.py in_file_name out_file_name"
	sys.exit(1)

try:
	f_read = open(sys.argv[1],"r")
	f_write = open(sys.argv[2],"w")
except IOError:
	print "file open failed"
	sys.exit(1)

user_idx = -1

max_user = -1
max_item = -1
max_rate = 0

while(True):
	str_line = f_read.readline()

	if len(str_line) <= 0:
		break

	if "|" in str_line: 
		str_list = str_line.strip("\n").split("|")
		#print str_list
		if len(str_list) == 2:
			user_idx = int(str_list[0])
			#print user_idx
			if user_idx > max_user:
				max_user = user_idx
	else:
		if user_idx == -1:
			continue
		else:
			str_list = str_line.strip("\n").split("\t")
			#if str_list[1] == "0":
				#str_list[1] = "1"
			f_write.write("%d %s %s\n"%(user_idx, str_list[0], str_list[1]))

			max_rate = max_rate + 1

			if int(str_list[0]) > max_item:
				max_item = int(str_list[0])


print "max_user: %d"%max_user
print "max_item: %d"%max_item
print "max_rate: %d"%max_rate



f_read.close()
f_write.close()

