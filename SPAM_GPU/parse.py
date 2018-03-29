#! /usr/bin/python

bnum = ['1024', '2048', '4192', '8192', '12288']
tnum = ['64', '128', '256', '512', '1024']

dset = [ '20', '40', '80', '100', '60', '120', '140', '160', '180', '200']
c='10'
t='5'
i='1'
algos= ['zero']
sup = '02'

for algo in algos:
	for d in dset:
		time = open('result/'+algo+'.time.d'+d+'c'+c+'t'+t+'i'+i+'.csv', 'w+')
		power = open('result/'+algo+'.power.d'+d+'c'+c+'t'+t+'i'+i+'.csv', 'w+')
		time.write(sup)
		power.write(sup)
		for thread in tnum:
			time.write(','+thread)
			power.write(','+thread)
		time.write('\n')
		power.write('\n')
		r = open('result/'+algo+'.d'+d+'c'+c+'t'+t+'i'+i+'s'+sup)
		for b in bnum:
			time.write(b)
			power.write(b)
			for thread in tnum:
				line = r.readline()
				while not line.startswith(sup):
					line = r.readline()
				while not line.startswith('bitmap'):
					print algo, d, b, thread
					line = r.readline()
				line = r.readline()
				if not line.startswith('total time for mining end:'):
					line = r.readline()
					time.write(',')
					power.write(',')
					break
				else:
					time.write(','+str(float(line.split("end:")[-1])/1000000))
				while not line.startswith('Mem usage:'):
					line = r.readline()
				line = r.readline()
				power.write(','+str(float(line)))
			power.write('\n')
			time.write('\n')
		power.write('\n')
		time.write('\n')
		r.close()
