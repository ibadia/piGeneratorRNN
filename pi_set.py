import numpy as np
from numpy import genfromtxt

f=open('pi-billion.txt','r')
a=f.read(1)
a=f.read(1)
a=f.read()
f.close()
maxlen=20
print (a[:30])
print len(a)
Question=[]
Expected=[]
samples=50000
for i in range(0,samples):
	toget=i+maxlen+1
	AA=a[:toget]
	AA=AA[i:]
	y=AA[len(AA)-1]
	x=AA[:len(AA)-1]
	x=str(x)
	y=str(y)
	Question.append(x)
	Expected.append(y)

f=open('pi_data.txt','w')
for i in range(0,samples):
	str1=''
	for item in Question[i]:
		str1+=str(item)+','
	str1+=str(Expected[i])
	str1+='\n'
	f.write(str1)
f.close()
	


