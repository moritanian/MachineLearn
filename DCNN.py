#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import glob
import os.path
from scipy import ndimage
import random
import copy
import sys

# global
# 本当は　インスタンスメンバにすべき
learn_weight = 0.02
sigmoid_a = 1.0
is_sigmoid = 0
layer_num = 3


img_s_flg = 0

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
		data['name'] = f
		data['arr'] = before_filter(data['img'])
		data_set.append(data)
	return data_set

# 学習に使用しないデータで確認
def test(test_data_set):
	global learn_data
	err_sum = 0.0
	success = 0
	data_num = len(test_data_set)

	print "test result"

	for data_set in test_data_set:
		arr = data_set['arr']
		ans = data_set['ans']
		name = data_set['name']
		err = forward_all(arr, ans)
		result = learn_data['result']
		err_sum += err
		pred = 1
		if(result < 0.5):
			pred = 0
		if(pred == ans):
			success += 1
		print name + " pred: " + str(result) + " ans: " + str(ans) 

	print "test result \n"
	print str(success) + "/" + str(data_num) 
	print "errs = " + str(err_sum/ data_num) 

# 確率的勾配降下法で学習
def learn(data_set, limit_err = 0.2):
	global layer_num, learn_data
	learn_data['err'] = 0
	learn_data['layer_num'] = layer_num
	layers = []
	for i in range(layer_num): 
		layer = {}
		layers.append(layer)

	layers[1]['W'] = np.random.rand(64, 64*64+1)/(64*64+1)
	layers[2]['W'] = np.random.rand(1, 65)/65

	learn_data['layers'] = layers
	err_sum = limit_err + 1.0

	loop_c = 0

	while err_sum > limit_err:
		crt_err = 0
		# shuffle
		size = len(data_set)
		index_list = range(size)
		random.shuffle(index_list)
		for index in index_list:
			#print " learn"
			#print index
			#print data_set[index]['name']
			err = learn_one(data_set[index])
			crt_err += err

		loop_c += 1
		err_sum = crt_err/len(index_list)
		print "err = "
		print err_sum
		if(loop_c > 100):
			sys.stderr.write('not solved ')
			break

		print "loop"
		print loop_c

	return err_sum

# 重み付けws を表示
def print_W():
	global layer_num
	for i in range(layer_num - 1):
		print  learn_data['layers'][i+1]['W']

# 一データに対し、学習し、誤差を返す
def learn_one(data):
	global learn_data
	# print data
	arr = data['arr']
	ans = data['ans']
	err = forward_all(arr, ans)
	layer_sum = learn_data['layer_num']
	
	# 逆伝播
	for i in range(learn_data['layer_num']-1):
		index = layer_sum - i - 1 # 2,1
		if(index == layer_sum - 1):  # 最後尾
			ws = learn_data['layers'][index]['W']
			xs = learn_data['layers'][index-1]['ys']
			ret = backward(ws, xs, learn_data['result'], np.array([ans]))
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

# 一回,回す
def forward_all(arr, ans):
	global learn_data
	layer_sum = learn_data['layer_num']
	learn_data['layers'][0]['ys'] =  copy.copy(arr) # 参照渡し　対策
	
	for i in range(learn_data['layer_num']-1):
		parent_layer = learn_data['layers'][i]
		layer = learn_data['layers'][i+1]
		arr = forward(parent_layer['ys'], layer['W'])
		layer['ys'] = copy.copy(arr)

	learn_data['result'] = learn_data['layers'][layer_sum -1]['ys'][0]
	err = learn_data['result'] - ans
	if(err < 0.0):
		err = -err

	#print "forward_all"
	#for i in range(layer_sum):
	#	print  learn_data['layers'][i]['ys']
	return err 

# 画像前処理
# 256*256 => 64*64 になる
def before_filter(img):
	global img_s_flg
	im = np.array(img.convert('L'))/256.0
	im = max_pooling(im)
	im = div_img(1.0, im)
	#im = laplacian(im)
	im = max_pooling(im)
	#im = div_img(1.0, im)	
	if img_s_flg == 0:
		Image.fromarray(im*256).show()
		img_s_flg = 1
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
	xs = np.append(arr, 1) # 定数項付加
	return apply_vector(np.dot(ws, xs), activation)	
	
# 行列の各要素に適応
def apply_each(mat, fun):
	# 要素が1つの時、2次元配列にならず、1次元になってしまう対策
	if mat.size == 1:
		return fun(mat)
	for y in xrange(mat[0].size):
		for x in xrange(mat.size/ mat[0].size):
			mat[x,y] = fun(mat[x,y])
	return mat

def get_matrix_width(mat):
	if(mat.size == 1):
		return 1
	if(mat.ndim == 1):
		return mat.size
	return mat[0].size

# vector の各要素に作用
def apply_vector(vec, fun):
	# 要素が1つの時, スカラーになってしまう
	if vec.size == 1:
		return fun(vec)
	for n in xrange(vec.size):
		vec[n] = fun(vec[n])
	return vec


def activation(x):
	global is_sigmoid
	if (is_sigmoid == 1):
		return sigmoid(x)
	return np.maximum(0,x)

# activation の微分を得る ただし引数は sigmoid(x) = y
def dif_activation(y):
	global sigmoid_a, is_sigmoid
	if (is_sigmoid == 1):
		return sigmoid_a * y * (1.0 - y)

	dif = 1.0
	if(y == 0.0):
		dif = 0.0
	return dif

def sigmoid(x, a = 1.0):
	return 1.0/(1.0+np.exp(-x*a))

# arr に変換
def convert_arr_from_img(img):
	size = img.size
	arr = img.reshape(1,size)
	arr = arr[0] # 1次元ベクトルに
	return arr

# correct_ys を計算し、ws を更新する 更新されたwsが返る(非破壊的　たぶん)
def backward(ws, xs, ys, correct_ys):
	delta_vec = (ys - correct_ys) * apply_vector(ys, dif_activation)
	return _backward(ws, xs, ys, delta_vec)

def backward_in_hidden_layer(ws, xs,  ys, child_ws, child_delta_vec):
	delta_vec = calc_delta_vec_in_hidden_layer(ys, child_ws, child_delta_vec)
	return _backward(ws, xs, ys, delta_vec)

# 引数　ws:重みづけ2次元ベクトル　xs:入力　ys:出力(vector)　correct_xs:正しい出力
# delta: vector 
def _backward(ws, xs, ys, delta_vec):
	global learn_weight
	del_mat = np.transpose( get_same_array(delta_vec, xs.size + 1))
	diag_arr = np.diag(np.append(xs,1))
	new_ws = ws - learn_weight * np.dot(del_mat, diag_arr) 
	return {'new_ws':new_ws, 'delta_vec':delta_vec}

# 隠れ層のデルタを計算
def calc_delta_vec_in_hidden_layer(ys, child_ws, child_delta_vec):
	global sigmoid_a
	w_size_in_child_ws = get_matrix_width(child_ws)
	if child_ws.ndim == 1: # 1次元の場合、転置できないため例外対応
		mod_child_ws = child_ws[0:w_size_in_child_ws-1].reshape([w_size_in_child_ws - 1, 1])
	else:
		mod_child_ws = (np.transpose(child_ws))[0:w_size_in_child_ws - 1] # 転置して定数項の部分を削除
	
	return np.dot(mod_child_ws, child_delta_vec) * sigmoid_a* ((np.ones(ys.size) - ys) * ys)

# [[arr],
#  [arr],
# [arr]] の配列を得る
def get_same_array(arr, h):
	if arr.ndim > 1 and arr.size != arr[0].size:
		print "warning!! invalid arr shape "
		sys.stderr.write(str(arr.shape))
	n_arr = np.zeros([h, arr.size])
	n_arr[0:h] = arr
	return n_arr


def main():
	
	# データセット
	f_paths_k =  glob.glob('./Pictures/teach/koala/*')
	f_paths_o = glob.glob('./Pictures/teach/other/*')
	data_set = make_data_set(f_paths_k, 1.0)
	data_set +=  make_data_set(f_paths_o, 0.0)

	print "all dataset = "
	print len(data_set)
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
		count += 1

	print learn_data_set
	learn(learn_data_set)
	print_W()
	test(test_data_set)
	#test(learn_data_set)
	return

if __name__ == '__main__':
	main()
