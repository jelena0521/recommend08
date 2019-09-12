#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 21:22:20 2019

@author: liujun
"""
import csv
import collections
import tensorflow as tf
from tensorflow.contrib.factorization import WALSModel
import numpy as np


k=10
n=10
reg=1e-1


        

Rating=collections.namedtuple('Rating',['user_id','item_id','rating','timestamp']) #命名一个元祖 及命名元祖中的元素
ratings=[]
with open('ratings.csv','r')as f:
    reader=csv.reader(f)
    next(reader) #读下一行
    for user_id,item_id,rating,timestamp in reader:
        ratings.append(Rating(user_id,item_id,float(rating),int(timestamp)))
ratings=sorted(ratings,key=lambda x:x.timestamp) #按时间排序
print('数据一共条数为{}'.format(len(ratings)))
       
    

users_from_idx=sorted(set(r.user_id for r in ratings)) #获得所有的user_id
users_from_idx=dict(enumerate(users_from_idx)) #{idx:userid}
users_to_idx=dict((user_id,idx) for idx,user_id in users_from_idx.items()) #{userid:idx}
items_from_idx=sorted(set(r.item_id for r in ratings)) #获取所有的item_id
items_from_idx=dict(enumerate(items_from_idx)) #{idx:userid}
items_to_idx=dict((item_id,idx) for idx,item_id in items_from_idx.items()) #{userid:idx}
indices=[(users_to_idx[r.user_id],items_to_idx[r.item_id]) for r in ratings] #[(userid,itemid)....]

sess=tf.InteractiveSession()

values=[r.rating for r in ratings] #所有的评价数据
n_rows=len(users_to_idx) #用户数
n_cols=len(items_to_idx) #物品数
shape=(n_rows,n_cols) 
P=tf.SparseTensor(indices,values,shape) #稀疏矩阵 行为userid 列为itemid  有值则为rating 没有则为0
print(P)

    
model=WALSModel(n_rows,n_cols,k,regularization=reg,unobserved_weight=0) #k是分解后的维度，n是迭代次数，reg为正则化权重
row_factors=tf.nn.embedding_lookup(  #取出行矩阵
                model.row_factors,
                tf.range(model._input_rows),
                partition_strategy='div')
col_factors=tf.nn.embedding_lookup(  #取出列矩阵
                model.col_factors,
                tf.range(model._input_cols),
                partition_strategy='div')
row_indices,col_indices=tf.split(P.indices,axis=1,num_or_size_splits=2) #获取稀疏矩阵中行和列的索引
gathered_row_factors=tf.gather(row_factors,row_indices) #根据索引取出行矩阵的值
gathered_col_factors=tf.gather(col_factors,col_indices) #根据索引取出列矩阵的值
approx_vals=tf.squeeze(tf.matmul(gathered_row_factors,gathered_col_factors,adjoint_b=True)) #删除维度为1的维度
P_approx=tf.SparseTensor(indices=P.indices,values=approx_vals,dense_shape=P.dense_shape) #将预测结果生成稀疏矩阵
E=tf.sparse_add(P,P_approx*(-1)) #矩阵相减
E2=tf.square(E) #平方
n_P=P.values.shape[0].value #行数
rmse_op=tf.sqrt(tf.sparse_reduce_sum(E2)/n_P) #loss
row_update_op=model.update_row_factors(sp_input=P)[1] #跟新分解矩阵的op
col_update_op=model.update_col_factors(sp_input=P)[1] 
model.initialize_op.run()
model.worker_init.run()
for _ in range(n):
    model.row_update_prep_gramian_op.run() #跟新行
    model.initialize_row_update_op.run()
    row_update_op.run()
    model.col_update_prep_gramian_op.run() #更新列
    model.initialize_col_update_op.run()
    col_update_op.run()
    print('RMSE:{:,.3f}'.format(rmse_op.eval())) #输出loss
user_factors=model.row_factors[0].eval()  
item_factors=model.col_factors[0].eval()
print('user factors shape:',user_factors.shape) #模型拆分以后的shape  用户数*K  
print('item factors shape:',item_factors.shape)  #模型拆分以后的shape  物品数*K 
    

   
c=collections.Counter(r.user_id for r in ratings) #根据userid的活跃度排序
user_id,n_ratings=c.most_common(1)[0] #找出评论最多的用户
print('评论最多的用户{}:{:,d}'.format(user_id,n_ratings))
r=next(r for r in reversed(ratings) if r.user_id==user_id and r.rating==5.0) #该用户最新的一条5分评分
print('该用户最后一条5分记录：',r)
i=users_to_idx[r.user_id] #找到该用户的userid索引
j=items_to_idx[r.item_id] #找到该物品的itemid的索引
u=user_factors[i] #找到分解矩阵中对应的行
v=item_factors[j] #找到分解矩阵中对应的行
p=np.dot(u,v) #得到预测的评分
print('approx rating:{:,.3f},diff={:,.3f},{:,.3%}'.format(p,r.rating-p,p/r.rating))

        

V=item_factors
user_p=np.dot(V,u) #将代表用户的行和每一列代表物品的列相乘 得到用户对每个物品的评价
user_items=set(ur.item_id for ur in ratings if ur.user_id==user_id) #找到该用户已评价的物品ID
user_ranking_idx=sorted(enumerate(user_p),key=lambda p:p[1]) #遍历预测该用户评价  并按评价值排序 index：预测分
user_ranking_row=((items_from_idx[j],p) for j,p in user_ranking_idx) #itemID：预测分
user_ranking=[(item_id,p) for item_id, p in user_ranking_row if item_id not in user_items] #筛选出未评价的部分
top10=user_ranking[:10] #取前十
print('top10:\n')
for k,(item_id,p) in enumerate(top10):
    print('{} {} {:,.3f}'.format(k+1,item_id,p))
        
        
        
        
        
        
        
        