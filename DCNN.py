#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import glob
import os.path
from scipy import ndimage
import random
import copy

# global
# 本当は　インスタンスメンバにすべき
learn_weight = 0.02
sigmoid_a = 1.0
is_sigmoid = 1
layer_num = 3

# 各種学習中のパラメータを入れる箱
# layer_num => int()
# err  => double()
# result => double()
# layers => {
#				[
#					ys => vector
#					W => matrix		
#				], ...		
#			}
#			
learn_data = {}

def make_data_set(paths, ans):
	data_set = []
	for f in paths:
		data = {}
		data['img'] = Image.open(f)
		data['ans'] = ans
		data_set.append(data)
	return data_set

# 確率的勾配降下法で学習
def learn(data_set, limit_err = 0.01):
	learn_data['err'] = 0
	learn_data['layer_num'] = layer_num
	layers = []
	for i in range(layer_num): 
		layer = {}
		layers.append()

		layers[1]['W'] = np.random.rand(64, 64*64+1)/(64*64+1)
		layers[2]['W'] = np.random.rand(1, 65)/65

	learn_data['layers'] = layers
	crt_err = limit_err + 1 

	loop_c = 0

	while crt_err > limit_err:
		crt_err = 0
		# shuffle
		size = len(data_set)
		index_list = range(size)
		random.shuffle(index_list)
		for index in index_list:
			err = learn_one(data_set[index])
			crt_err += err

		loop_c += 1
		if(loop_c > 1000):
			break

	return crt_err

# 一データに対し、学習し、誤差を返す
def learn_one(data):
	img = data['img']
	arr = before_filter(img)
	layer_sum = learn_data['layer_num']

	learn_data['layers'][0]['ys'] =  copy.copy(arr) # 参照渡し　対策
	
	# 回路を前へ回す
	for i in range(learn_data['layer_num']-1):
		parent_layer = learn_data['layers'][i]
		layer = learn_data['layers'][i+1]
		arr = forward(parent_layer['ys'], layer['W'])
		layer['ys'] = copy.copy(arr)

	learn_data['result'] = learn_data['layers'][layer_sum -1]['ys'][0]
	err = learn_data['result'] - data['ans']

	# 逆伝播
	for i in range(learn_data['layer_num']-1):
		index = layer_sum - i - 1 # 2,1
		if(index == layer_sum - 1):  # 最後尾
			ws = learn_data['layers'][index]['W']
			xs = learn_data['layers'][index]['ys']
			ret = backward(ws, xs, learn_data['result'])
			new_ws = ret['new_ws']
			delta_vec = ret['delta_vec']
		else:
			ws = learn_data['layers'][index]['W']
			xs = learn_data['layers'][index - 1]['ys']
			ys = learn_data['layers'][index]['ys']
			child_ws = learn_data['layers'][index + 1]['W']
			child_delta_vec = copy.copy(delta_vec) # とりあえずコピーしておく　必要ないかも
			ret = backward_in_hidden_layer(ws, xs,  ys, child_ws, child_delta_vec)
			learn_data['layers'][index+1]['W'] = copy.copy(new_ws) # 先に前のぶんを更新しておく
			new_ws = ret['new_ws']
			delta_vec = ret['delta_vec']


	learn_data['layers'][1]['W'] = copy.copy(new_ws)
	return err

# 画像前処理
# 256*256 => 64*64 になる
def before_filter(img):
	im = np.array(im.convert('L'))/256.0
	im = max_pooling(im)
	im = div_img(1.0, im)
	im = max_pooling(im)
	im = div_img(1.0, im)	
	arr = convert_arr_from_img(im)
	return arr

# 微分フィルタ
def div_img(w, img):
	fil = np.array( [[-w , 0, w],
       			[-w, 0, w],
       			[-w, 0, w]])
	con = apply_filter(fil, img) #畳み込み
	return con

# 微分フィルタ
def div_img2(w, img):
	fil = np.array( [[-w , -w, -w],
       			[0, 0, 0],
       			[w, w, w]])
	con = apply_filter(fil, img) #畳み込み
	return con

def laplacian(img):
	fil = np.array( [[1.0 , 1.0, 1.0],
       			[1.0, -8.0, 1.0],
       			[1.0, 1.0, 1.0]])
	con = apply_filter(fil, img) #畳み込み
	return con

def max_pooling(img):
	h = img.size/2/img[0].size
	w = img[0].size/2
	r_img = np.zeros((h,w))
	for x in range(h):
		for y in range(w):
			r_img[x][y] = max(img[x*2][y*2], img[x*2+1][y*2], img[x*2][y*2+1], img[x*2+1][y*2+1]) 
	return r_img



def gaussian_img(img):
	fil = np.array( [[1.0 , 2.0, 1.0],
       			[2.0, 4.0, 2.0],
       			[1.0, 2.0, 1.0]])/16.0
	con = apply_filter(fil, img) #畳み込み
	return con

def apply_filter(fil, img):
	return  ndimage.convolve(img, fil)


# 重み配列(2次元)　でarrayから値を計算
def forward(arr, ws):
	applied = apply_each(np.dot(ws, arr), activation)	
	return np.append(applied, 1)

# 行列の各要素に適応
def apply_each(mat, fun):
	# 要素が1つの時、2次元配列にならず、1次元になってしまう対策
	if mat.size == 1:
		return fun(mat)
	for y in xrange(mat[0].size):
		for x in xrange(mat.size/ mat[0].size):
			mat[x,y] = fun(mat[x,y])
	return mat

# vector の各要素に作用
def apply_vector(vec, fun):
	for n in xrange(vec.size):
		vec[n] = fun(vec[n])
	return vec


def activation(x):
	if (is_sigmoid == 1):
		return sigmoid(x)
	return np.maximum(0,x)

# activation の微分を得る ただし引数は sigmoid(x) = y
def dif_activation(y):
	if (is_sigmoid == 1):
		return sigmoid_a * y * (1.0 - y)

	dif = 1.0
	if(y == 0.0):
		dif = 0.0
	return dif

def sigmoid(x, a = 1.0):
	return 1.0/(1.0+np.exp(-x*a))

# 定数係数を付加したarr に変換
def convert_arr_from_img(img):
	size = img.size
	arr = img.reshape(size, 1)
	arr = np.append(arr,[-1.0])
	#arr = arr.reshape(arr.size, 1)
	arr = np.c_[arr]
	return arr

# correct_ys を計算し、ws を更新する 更新されたwsが返る(非破壊的　たぶん)
def backward(ws, xs, ys):
	delta_vec = (ys - correct_ys) * apply_vector(ys, dif_activation)
	return _backward(ws, xs, ys, delta_vec)

def backward_in_hidden_layer(ws, xs,  ys, child_ws, child_delta_vec):
	delta_vec = calc_delta_vec_in_hidden_layer(ys, child_ws, child_delta_vec)
	return _backward(ws, xs, ys, delta_vec)

# 引数　ws:重みづけ2次元ベクトル　xs:入力　ys:出力(vector)　correct_xs:正しい出力
# delta: vector 
def _backward(ws, xs, ys, delta_vec):
	del_mat = np.transpose( get_same_array(delta_vec, xs.size))
	diag_arr = np.diag(xs)
	new_ws = ws - learn_weight * np.dot(del_mat, diag_arr) 
	return {'new_ws' => new_ws, 'delta_vec' => delta_vec}

# 隠れ層のデルタを計算
def calc_delta_vec_in_hidden_layer(ys, child_ws, child_delta_vec):
	return np.dot(child_ws, child_delta_vec) * sigmoid_a* ((np.ones(ys.size) - ys) * ys)

# [[arr],
#  [arr],
# [arr]] の配列を得る
def get_same_array(arr, h):
	n_arr = zeros([h, arr.size])
	n_arr[0:h] = arr
	return n_arr


def main():
	
	# データセット
	f_paths_k =  glob.glob('./Pictures/teach/koala/*')
	f_paths_o = glob.glob('./Pictures/teach/other/*')
	data_set = make_data_set(f_paths_k, 1.0)
	data_set.append(make_data_set(f_paths_o, 0.0))

	# 無作為に学習データと検証データを選別
	learn_data_set = []
	test_data_set = []
	data_sum = len(data_set)
	num_list = range(data_sum)
	random.shuffle(num_list)


	# 学習に使うのは8割
	learn_sum = int(data_sum*0.8)
	count = 0
	for index in num_list:
		if count < learn_sum:
			learn_data_set.append(data_set[index])
		else:
			test_data_set.append(data_set[index])


	return

if __name__ == '__main__':
	main()
