# a script to test PyEEG on real EEG data
# just put this script together with the data folders
# for paper PyEEG:A Open Source Python Module for EEG/MEG Feature Extraction
# 2010/10/18  Xin Lin
#

# Sets A and B consist of segments taken from five healthy volunteers with eyes open(A) and eye close(B)
# Sets C, D, non-seizure zone
# Set E seizure zone

import sys,string
from pyeeg import *
from pylab import *
from numpy import *

""" This is a script for testing PyEEG on the data sets from m Klinik f\"ur Epileptologie, Universit\"at Bonn, German.
	
	To try this script, just put it in the same directory with the five data folders.
	
	Basic useage: python script FEATURE_NAME
	
	Seprate the FEATURE_NAME with space, if there is no FETURE_NAME, this script will compute all the features.
		
	Examples 
	--------	
	python script dfa                compute the dfa feature
	
	python script dfa spect       compute the dfa and spectral_entropy feature
	
	python script                     compute all the features

	Here list the feature abbreviations 
	
	'spect'  ======> spectral entropy
	'bp'      ======> bin_power
	'hurst'  ======> hurst
	'dfa'     ======> dfa
	'hjorth' ======> hjorth
	'pfd'     ======> pfd
	'apen'   ======> ap_en
	'hfd'     ======> hfd
	'fi'        ======> fisher_information
	'svd'     ======> svd_entropy
	
"""
#### definition of feature plot function ###########

def feature_plot(IDX):

	AXIS = array([[-0.5,4.5,0.12,0.31],
						[-0.5,4.5,0.32,0.98],
						[-0.5,4.5,0.02,0.12],
						[-0.5,4.5,0.11,0.25],
						[-0.5,4.5,0, 1],
						[-0.5,4.5,0.52,0.69],
						[-0.5,4.5,0.574,0.609],
						[-0.5,4.5,0, 1],
						[-0.5,4.5,0.67,0.82],
						[-0.5,4.5,1.92,2.26]])

	N = 5
	NAME_MAP = {'1':'AP', '2':'DFA', '3':'FI', '4':'HFD', '5':'Hjorth', '6':'HURST', '7':'PFD', '8':'PSI', '9':'SPECT', '10':'SVD'}
	for i in xrange(1, len(IDX)):
		if IDX[i] == 1:
			NAME = NAME_MAP[str(i)]
			f1=open('Z/'+NAME+'.A.txt','r')
			A=[]
			for line in f1.readlines():
				A.append(map(double, line.split()))

			f2=open('O/'+NAME+'.B.txt','r')
			B=[]
			for line in f2.readlines():
				B.append(map(double, line.split()))

			f3=open('N/'+NAME+'.C.txt','r')
			C=[]
			for line in f3.readlines():
				C.append(map(double, line.split()))

			f4=open('F/'+NAME+'.D.txt','r')
			D=[]
			for line in f4.readlines():
				D.append(map(double, line.split()))

			f5=open('S/'+NAME+'.E.txt','r')
			E=[]
			for line in f5.readlines():
				E.append(map(double, line.split()))


			A = array(A)
			B = array(B)
			C = array(C)
			D = array(D)
			E = array(E)
			
			x = range(0,len(A))
			if NAME == 'Hjorth':
				Mobmean = [mean(A[:,0]),mean(B[:,0]),mean(C[:,0]),mean(D[:,0]),mean(E[:,0])]
				Mobstd = [var(A[:,0]),var(B[:,0]),var(C[:,0]),var(D[:,0]),var(E[:,0])]
				maxcom=max(max(A[:,1]),max(B[:,1]),max(C[:,1]),max(D[:,1]),max(E[:,1]))
				mincom=min(min(A[:,1]),min(B[:,1]),min(C[:,1]),min(D[:,1]),min(E[:,1]))
				A[:,1]=(A[:,1]-mincom)/(maxcom-mincom)
				B[:,1]=(B[:,1]-mincom)/(maxcom-mincom)
				C[:,1]=(C[:,1]-mincom)/(maxcom-mincom)
				D[:,1]=(D[:,1]-mincom)/(maxcom-mincom)
				E[:,1]=(E[:,1]-mincom)/(maxcom-mincom)
				Commean = [mean(A[:,1]),mean(B[:,1]),mean(C[:,1]),mean(D[:,1]),mean(E[:,1])]
				Comstd = [var(A[:,1]),var(B[:,1]),var(C[:,1]),var(D[:,1]),var(E[:,1])]
		
				figure(i, figsize=(12,6))
				subplot(121)
				ind = arange(N)  # the x locations for the groups
				width = 0.5       # the width of the bars
				p2 = errorbar(ind, Mobmean, Mobstd,marker='s',linestyle='-',linewidth=0.5,markersize=0.1)
				axis([-0.5,4.5,0.0025,0.0060])
				xticks(ind, ('A', 'B', 'C', 'D', 'E'),size=20 )
				title('(a) Hjorth Mobility',size=20)
				subplot(122)
				p2 = errorbar(ind, Commean, Comstd,marker='s',linestyle='-',linewidth=0.5,markersize=0.1)
				xticks(ind, ('A', 'B', 'C', 'D', 'E'),size=20)
				xlim(-width,len(ind))
				title('(b) Hjorth Complexity',size=20)
				savefig('Hjorth.png')
					
			elif NAME == 'PSI':			### bin power case
		
				MA = mean(A, 0)
				MB = mean(B, 0)
				MC = mean(C, 0)
				MD = mean(D, 0)
				ME = mean(E, 0)
		
				figure(i, figsize=(30,6))
				subplot(151)
				plot(MA, 'b')
				#import code;code.interact(local=locals())
				axis([-1,43,0,350000])
				grid(True)
				title('A', size=20)

				subplot(152)
				plot(MB, 'c')
				axis([-1,43,0,500000])
				grid(True)
				title('B', size=20)

				subplot(153)
				plot(MC, 'm')
				axis([-1,43,0,600000])
				grid(True)
				title('C', size=20)

				subplot(154)
				plot(MD, 'r')
				axis([-1,43,0,750000])
				grid(True)
				title('D', size=20)

				subplot(155)
				plot(ME, 'y')
				axis([-1,43,0,2500000])
				grid(True)
				title('E', size=20)
				savefig('PSI.png')
	
			else:				## other cases
	
				Mean = [mean(A),mean(B),mean(C),mean(D),mean(E)]
				Std = [var(A),var(B),var(C),var(D),var(E)]

				figure(i, figsize=(5,5))
				ind = arange(N)  # the x locations for the groups
				width = 0.5       # the width of the bars		
				p2 = errorbar(ind, Mean, Std,marker='s',linestyle='-',linewidth=0.5,markersize=2)
				axis(AXIS[i-1][:])
				xticks(ind, ('A', 'B', 'C', 'D', 'E'),size=20)
				title(NAME,size=20)
				savefig(NAME + '.png')
	show()
	
####### Begin the main part #####################
####### Define the parameters ##############
### Note if you change these default parameter values, you may need to adjust the axis interval when plotting

DIM = 10
TAU = 4
Kmax = 5
Fs = 173
Band = [2*i+1 for i in xrange(0, 43)]		## 0.5~85 Hz
SET_MAP = {'A':'Z', 'B':'O', 'C':'N', 'D':'F','E':'S'}
FEA_IDX = {'spect':0, 'bp':0, 'hurst':0, 'dfa':0, 'hjorth':0, 'pfd':0, 'apen':0, 'hfd':0, 'fi':0, 'svd':0}
FEA_MAP = {'apen':'1', 'dfa':'2', 'fi':'3', 'hfd':'4', 'hjorth':'5', 'hurst':'6', 'pfd':'7', 'bp':'8', 'spect':'9', 'svd':'10'}
ALL = 0
FLAG = zeros(11)

FEA_NUM = len(sys.argv)
if FEA_NUM == 1:
	ALL = 1
	FLAG = ones(11)
else:
	for i in xrange(1, FEA_NUM):
		tmp = sys.argv[i].lower()
		FEA_IDX[tmp] = 1
		FLAG[int(FEA_MAP[tmp])] = 1

##### Feature Extraction part################
for i in xrange(5):
	print 'Begin the %dth SET' % (i+1)
	SET_NO = 65 + i;
	SET_NAME = chr(SET_NO);
	SET_NAME = SET_MAP[SET_NAME]
	for j in xrange(1, 101):
		print 'Begin the %dth SEGMENT of SET %d' % (j,i+1)
		if j < 10:
			FILE_NAME = '00' + str(j)
		elif j < 100:
			FILE_NAME = '0' + str(j)
		else:
			FILE_NAME = str(j)
		if i == 2:
			FILE_NAME = SET_NAME + FILE_NAME + '.TXT'
		else:
			FILE_NAME = SET_NAME + FILE_NAME + '.txt'

		FILE_DIR = SET_NAME + '/' + FILE_NAME
		fid = open(FILE_DIR, 'r')
		tmp = fid.readlines()
		DATA = [float(k) for k in tmp]
	####### Methods ######################	
		if (ALL or FEA_IDX['spect']):
			print 'spectral entropy...',
			RESULT_DIR = SET_NAME + '/' + 'SPECT.' + chr(SET_NO) + '.txt'			
			fod = open(RESULT_DIR, 'a')
			result = spectral_entropy(DATA, Band, Fs)
			fod.write('%f\n' % float(result))
			
		if (ALL or FEA_IDX['bp']):
			print 'PSI ...',
			RESULT_DIR = SET_NAME + '/' +'PSI.' + chr(SET_NO) + '.txt'
			fod = open(RESULT_DIR, 'a')
			result = bin_power(DATA, Band, Fs)
			for k in result[0]:
				fod.write('%f\t' % float(k))
			fod.write('\n')
			RESULT_DIR = SET_NAME + '/' +'RIR.' +chr(SET_NO) + '.txt'
			fod = open(RESULT_DIR, 'a')
			for k in result[1]:
				fod.write('%f\t' % float(k))
			fod.write('\n')	
	
		if (ALL or FEA_IDX['hurst']):
			print 'Hurst Exponent...',
			RESULT_DIR = SET_NAME + '/' + 'HURST.' + chr(SET_NO) + '.txt'
			fod = open(RESULT_DIR, 'a')
			result = hurst(DATA)
			if not isnan(result):
				fod.write('%f\n' % float(result))
		
		if (ALL or FEA_IDX['dfa']):
			print 'DFA...',
			RESULT_DIR = SET_NAME + '/' + 'DFA.' + chr(SET_NO) + '.txt'
			fod = open(RESULT_DIR, 'a')
			#import code;code.interact(local=locals())
			result = dfa(DATA)
			fod.write('%f\n' % float(result))
		
		if (ALL or FEA_IDX['hjorth']):
			print 'Hjorth...',
			RESULT_DIR = SET_NAME + '/' + 'Hjorth.' + chr(SET_NO) + '.txt'
			fod = open(RESULT_DIR, 'a')
			result = hjorth(DATA)
			fod.write('%f\t%f\n' % (float(result[0]),float(result[1])))

		if (ALL or FEA_IDX['pfd']):
			print 'PFD...',
			RESULT_DIR = SET_NAME + '/' + 'PFD.' + chr(SET_NO) + '.txt'
			fod = open(RESULT_DIR, 'a')
			result = pfd(DATA)
			fod.write('%f\n' % float(result))
		
		if (ALL or FEA_IDX['apen']):
			print 'approximate entropy...',
			R = std(DATA) * 0.3
			RESULT_DIR = SET_NAME + '/' + 'AP.' + chr(SET_NO) + '.txt'
			fod = open(RESULT_DIR, 'a')
			result = ap_entropy(DATA, DIM, R)
			fod.write('%f\n' % float(result))
		
		if (ALL or FEA_IDX['hfd']):
			print 'HFD...',
			RESULT_DIR = SET_NAME + '/' + 'HFD.' + chr(SET_NO) + '.txt'
			fod = open(RESULT_DIR, 'a')
			result = hfd(DATA, Kmax)
			fod.write('%f\n' % float(result))
		
		if (ALL or FEA_IDX['fi']):
			print 'fisher information...',
			M = embed_seq(DATA, TAU, DIM)
			W = svd(M, compute_uv=0)
			W /= sum(W)		
			RESULT_DIR = SET_NAME + '/' + 'FI.' + chr(SET_NO) + '.txt'
			fod = open(RESULT_DIR, 'a')		
			result = fisher_info(DATA, TAU, DIM, W)
			fod.write('%f\n' % float(result))
		
		if (ALL or FEA_IDX['svd']):
			print 'SVD entropy...'
			M = embed_seq(DATA, TAU, DIM)
			W = svd(M, compute_uv=0)
			W /= sum(W)
			RESULT_DIR = SET_NAME + '/' + 'SVD.' + chr(SET_NO) + '.txt'
			fod = open(RESULT_DIR, 'a')
			result = svd_entropy(DATA, TAU, DIM, W)
			fod.write('%f\n' % float(result))

###### Plot part################
feature_plot(FLAG)
