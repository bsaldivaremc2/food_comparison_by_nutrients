import pandas as pd
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
class ModelScorePlot:
	def normNp(self,inp,t='zo'):
		maxv=inp.max()
		minv=inp.min()
		return (inp-minv)/(maxv-minv)
	def simColsRef(self,baseList,df2):
	    cols1=baseList
	    cols2=list(df2.columns.values)
	    colsList=list()
	    for col in cols1:
        	if col in cols2:
	            colsList.append(col)
	    return colsList.copy()
	def plotTrainTest(self,train,test,rangeX,t='Title',figsize=(12,6),dpi=80,xlabel='Optimization parameter',ylabel='score',plot=True):
		"""
		Calculate the average of the *train_score* and *test_score* outputs of a validation_curve. 
		Then the function finds the maximum score in the test_score for the :param xlabel: .
		Outputs a dictionary with the value of the :param xlabel: with key *best_param*, the *train_score* 
		associated to it with key *train_score* and the *test_score* with the key *test_score*
		"""
		trainMean=train.mean(axis=1)
		testMean=test.mean(axis=1)
		maxi=np.argmax(testMean)
		maxn=np.max(testMean)
		if plot==True:
			plt.figure(figsize=figsize,dpi=dpi)
			plt.plot(rangeX,trainMean,label='train')
			plt.plot(rangeX,testMean,label='test')
			plt.scatter(rangeX[maxi],maxn,c='red')
			plt.text(rangeX[maxi],maxn,"Max param: "+str(rangeX[maxi])+" Val: "+str(maxn))
			plt.ylabel(ylabel)
			plt.xlabel(xlabel)
			plt.title(t)
			plt.legend()
			plt.show()
		return {'best_param':rangeX[maxi],'train_score':trainMean[maxi],'test_score':testMean[maxi]}
	def combineDfs(self,iDfL,typesL,colorL,figSize=(50,50),iDpi=80,plotText=True):
		dfxL=list()
		for df,t in zip(iDfL,typesL):
			dft=df.copy()
			dft['type']=t
			dfxL.append(dft)
		ref_df=list(dfxL[0].columns.values)
		for i in range(1,len(dfxL)):
			ref_df=simColsRef(baseList=ref_df,df2=dfxL[i])
		ref_df_nt=ref_df.copy()
		ref_df_nt.remove('type')
		dfxSL=list()
		for df in dfxL:
			dfxSL.append(df[ref_df].copy())
		dfx_u=pd.concat(dfxSL)
		dfx_u_nt=dfx_u[ref_df_nt]
		tsne=TSNE()
		tsne_cords=tsne.fit_transform(dfx_u_nt)
		labels=list(dfx_u.index.values)
		typeL=list(dfx_u['type'].values)
		tsne_cords_n=normNp(tsne_cords)
		tsnen_df=pd.DataFrame(tsne_cords_n,index=dfx_u.index)
		plt.figure(figsize=figSize,dpi=iDpi)
		for t,c in zip(typesL,colorL):
			plt_df=tsnen_df[dfx_u['type']==t]
			plt.scatter(plt_df[0],plt_df[1],c=c,label=t)
		if plotText==True:
			for i in range(0,len(tsnen_df)):
				plt.text(tsnen_df.iloc[i,0],tsnen_df.iloc[i,1],list(tsnen_df.index.values)[i])
		plt.legend()
		plt.show()    
		return {'tsne':tsnen_df.copy(),'df':dfx_u.copy()}
	def scoreModelListDf(self,iScoreL,trainW=1,testW=2):
		"""
		Function used to merge the scores of various models
		"""
		svc_model_df=pd.DataFrame(iScoreL)
		svc_model_df['weighted_score']=(svc_model_df['test_score']*testW+svc_model_df['train_score']*trainW)/(trainW+testW)
		return svc_model_df.sort_values(by='weighted_score',ascending=False).copy()
	def svcScores(self,Xn,y,cv=5,param_name='C',max_iter=5000,degrees=(2,6,1),paramRange=(1,10,1),trainW=1,testW=2,clfArg={},plot=False):
		"""
		Perform the validation_curve function using Support Vector classifier (SVC)
		and get the best param value based on the highest test_score. 
		cv indicates the cross validation k-fold. Default param to optimize is C. 
		max_iter is the maximum number of iterations for SVC.
		degrees=(a,b,c) is the range of degrees to evaluation when the SVC has poly as kernel.
		 a start degree, b end degree, c step.
		paramRange=(a,b,c) is the range to evaluate the param_name.
		After the function gets the best param value, associated test_score and 
		train_score are used to calculated a weighted_score.
		trainW and testW are the weights used to calculated a 
		weighted_score=test_score*testW+train_score*trainW)/(testW+trainW).
		clfArg is a dictionary to add any additional parameters to the SVC. 
		To see how the best score is collected set plot=True.
		"""
		kernels=['linear', 'poly', 'rbf', 'sigmoid']
		degrees=np.arange(degrees[0],degrees[1],degrees[2])
		model_scores=list()
		param_range=np.arange(paramRange[0],paramRange[1],paramRange[2])
		svc=SVC(**clfArg)
		svc.max_iter=max_iter
		for kernel in kernels:
			title='SVC '+kernel
			svc.kernel=kernel
			if kernel=='poly':
				for degree in degrees:
					svc.degree=degree
					dtitle=title+str(degree)
					train_sc, test_sc = validation_curve(svc,Xn,y,param_name=param_name,param_range=param_range,cv=cv)
					param_score= self.plotTrainTest(train_sc,test_sc,param_range,t=dtitle,xlabel=param_name,plot=plot)
					scoreDic={'model':dtitle,'param_name':param_name}
					scoreDic.update(param_score)
					model_scores.append(scoreDic.copy())
			else:
				train_sc, test_sc = validation_curve(svc,Xn,y,param_name=param_name,param_range=param_range,cv=cv)
				param_score=self.plotTrainTest(train_sc,test_sc,param_range,t=title,xlabel=param_name,plot=plot)
				scoreDic={'model':title,'param_name':param_name}
				scoreDic.update(param_score)
				model_scores.append(scoreDic.copy())
		return self.scoreModelListDf(model_scores,trainW=trainW,testW=testW)
	def kncScores(self,Xn,y,cv=5,param_name='n_neighbors',paramRange=(1,10,1),trainW=1,testW=2,title='KNC',clfArg={},plot=False):
		"""
		Perform the validation_curve function using K neighbors classifier (KNC)
		 and get the best param value based on the highest test_score. 
		cv indicates the cross validation k-fold. Default param_name to optimize is n_neighbors. 
		paramRange=(a,b,c) is the range to evaluate the param_name. a start degree, b end degree, c step.
		After the function gets the best param value, associated test_score and
		 train_score are used to calculated a weighted_score.
		trainW and testW are the weights used to calculated a 
		weighted_score=test_score*testW+train_score*trainW)/(testW+trainW).
		clfArg is a dictionary to add any additional parameters to the KNC. 
		To see how the best score is collected set plot=True.
		"""
		clf = KNC(**clfArg)
		model_scores=list()
		param_range=np.arange(paramRange[0],paramRange[1],paramRange[2])
		train_sc, test_sc = validation_curve(clf,Xn,y,param_name=param_name,param_range=param_range,cv=cv)
		param_score= self.plotTrainTest(train_sc,test_sc,param_range,t=title,xlabel=param_name,plot=plot)
		scoreDic={'model':title,'param_name':param_name}
		scoreDic.update(param_score)
		model_scores.append(scoreDic.copy())
		return self.scoreModelListDf(model_scores,trainW=trainW,testW=testW)
	def dtcScores(self,Xn,y,cv=5,param_name='max_depth',paramRange=(1,10,1),trainW=1,testW=2,title='Decision Tree classifier',clfArg={},plot=False):
		"""
		Perform the validation_curve function using Decision Tree classifier (DTC)
		 and get the best param value based on the highest test_score. 
		cv indicates the cross validation k-fold. Default param to optimize is max_depth. 
		paramRange=(a,b,c) is the range to evaluate the param_name. a start degree, b end degree, c step.
		After the function gets the best param value, associated test_score and 
		train_score are used to calculated a weighted_score.
		trainW and testW are the weights used to calculated a 
		weighted_score=test_score*testW+train_score*trainW)/(testW+trainW).
		clfArg is a dictionary to add any additional parameters to the DTC. 
		To see how the best score is collected set plot=True. 
		The function calculates the scores for the DTC criterions gini and entropy.
		"""
		clf=DTC(**clfArg)
		model_scores=list()
		param_range=np.arange(paramRange[0],paramRange[1],paramRange[2])
		criterions=['gini','entropy']
		for criterion in criterions:
			dtitle=title+" "+criterion
			clf.criterion=criterion
			train_sc, test_sc = validation_curve(clf,Xn,y,param_name=param_name,param_range=param_range,cv=cv)
			param_score=self.plotTrainTest(train_sc,test_sc,param_range,t=dtitle,xlabel=param_name,plot=plot)
			scoreDic={'model':dtitle,'param_name':param_name}
			scoreDic.update(param_score)
			model_scores.append(scoreDic.copy())
		return self.scoreModelListDf(model_scores,trainW=trainW,testW=testW)
	def rfcScores(self,Xn,y,cv=5,param_name='max_depth',estimatorsRange=(10,11,1),paramRange=(1,10,1),trainW=1,testW=2,title='Randorm Forest classifier',clfArg={},plot=False):
		"""
		Perform the validation_curve function using Random Forest classifier (RFC)
		 and get the best param value based on the highest test_score. 
		cv indicates the cross validation k-fold. Default param to optimize is max_depth. 
		paramRange=(a,b,c) is the range to evaluate the param_name. a start degree, b end degree, c step.
		estimatorsRange=(a,b,c) is the range to evaluate the number of estimators (n_estimators). 
		After the function gets the best param value, associated test_score and train_score
		 are used to calculated a weighted_score.
		trainW and testW are the weights used to calculated a 
		weighted_score=test_score*testW+train_score*trainW)/(testW+trainW).
		clfArg is a dictionary to add any additional parameters to the RFC. 
		To see how the best score is collected set plot=True. 
		The function calculates the scores for the RFC criterions gini and entropy.
		"""
		clf=RFC(**clfArg)
		model_scores=list()
		param_range=np.arange(paramRange[0],paramRange[1],paramRange[2])
		e_range=np.arange(estimatorsRange[0],estimatorsRange[1],estimatorsRange[2])
		criterions=['gini','entropy']
		for criterion in criterions:
			clf.criterion=criterion
			for e in e_range:
				clf.n_estimators=e
				dtitle=title+". Criterion: "+criterion+". Estimators: "+str(e)
				train_sc, test_sc = validation_curve(clf,Xn,y,param_name=param_name,param_range=param_range,cv=cv)
				param_score=self.plotTrainTest(train_sc,test_sc,param_range,t=dtitle,xlabel=param_name,plot=plot)
				scoreDic={'model':dtitle,'param_name':param_name}
				scoreDic.update(param_score)
				model_scores.append(scoreDic.copy())
		return self.scoreModelListDf(model_scores,trainW=trainW,testW=testW)
	def abcScores(self,Xn,y,cv=5,param_name='n_estimators',paramRange=(1,10,1),trainW=1,testW=2,title='Adaboost classifier',clfArg={},plot=False):
		"""
		Perform the validation_curve function using Adaboost classifier (ABC) 
		and get the best param value based on the highest test_score. 
		cv indicates the cross validation k-fold. Default param to optimize is max_depth. 
		paramRange=(a,b,c) is the range to evaluate the param_name. a start degree, b end degree, c step.
		After the function gets the best param value, associated test_score and train_score 
		are used to calculated a weighted_score.
		trainW and testW are the weights used to calculated a 
		weighted_score=test_score*testW+train_score*trainW)/(testW+trainW).
		clfArg is a dictionary to add any additional parameters to the ABC. 
		To see how the best score is collected set plot=True. 
		"""
		clf=ABC(**clfArg)
		model_scores=list()
		param_range=np.arange(paramRange[0],paramRange[1],paramRange[2])
		train_sc, test_sc = validation_curve(clf,Xn,y,param_name=param_name,param_range=param_range,cv=cv)
		param_score=self.plotTrainTest(train_sc,test_sc,param_range,t=title,xlabel=param_name,plot=plot)
		scoreDic={'model':title,'param_name':param_name}
		scoreDic.update(param_score)
		model_scores.append(scoreDic.copy())
		return self.scoreModelListDf(model_scores,trainW=trainW,testW=testW)
	def modelsCalculation(self,Xn,y,modelsL=['knc','svc','dtc','rfc','abc'],knc={},svc={},dtc={},rfc={},abc={}):
		"""
		Calculate the best param for various models. 
		modelsL is a list with the available models to evaluate the best param.
			knc: KNeighborsClassifier
			svc Support Vector classifier.
			dtc: DecisionTreeClassifier.
			rfc: RandomForestClassifier
			abc: AdaBoostClassifier
		The dictiories with the same names are to set the parameters for each Score function above.
		It returns a pandas DataFrame with the scores and best param found sorted with the best classifier on top.
		"""
		modelsScores=list()
		if 'knc' in modelsL:
			self.modelsScores.append(kncScores(Xn,y,**knc))
		if 'svc' in modelsL:
			self.modelsScores.append(svcScores(Xn,y,**svc))
		if 'dtc' in modelsL:
			self.modelsScores.append(dtcScores(Xn,y,**dtc))
		if 'rfc' in modelsL:
			self.modelsScores.append(rfcScores(Xn,y,**rfc))
		if 'abc' in modelsL:
			self.modelsScores.append(abcScores(Xn,y,**abc))
		mod_df=pd.concat(modelsScores).sort_values(by='weighted_score',ascending=False)
		return mod_df.copy()