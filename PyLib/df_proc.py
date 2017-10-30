import pandas as pd
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
def normNp(inp,t='zo'):
    maxv=inp.max()
    minv=inp.min()
    return (inp-minv)/(maxv-minv)
def simColsRef(baseList,df2):
    cols1=baseList
    cols2=list(df2.columns.values)
    colsList=list()
    for col in cols1:
        if col in cols2:
            colsList.append(col)
    return colsList.copy()
def combineDfs(iDfL,typesL,colorL,figSize=(50,50),iDpi=80,plotText=True):
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
