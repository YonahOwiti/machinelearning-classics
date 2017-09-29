'''
	Created on Sep 29, 2017

	@author: Varela

	ref: Blum. Foundations of Data Science Exercise 3.27, using TruncatedSVD
			We 

'''

import numpy as np 

from utils import get_lena
# from sklearn.decomposition import TruncatedSVD

def svd_compress_gs(gsimg, r=3, verbose=True):
	'''
		Applies SVD to a gray scale matrix

		gsimg .: gets a gray scale( 2D matrix where each pixel is represented by a 1 byte 0..255 ) and applies truncated SVD

	'''	
	u, s, v = np.linalg.svd(gsimg, full_matrices=True)

	if verbose: 
		print 'shapes: u:',u.shape,'\ts: ',s.shape,'\tv:\t',v.shape
	# svd = TruncatedSVD(n_components=r)
	# newimg = svd.fit_transform(gsimg)	
	# explained_variance = svd.explained_variance_
	# explained_variance_ratio = svd.explained_variance_ratio_	

	return newimg, explained_variance, explained_variance_ratio


def svd_compress_rgb(rgbimg, r=3):
	'''
		reuses svd_compress_rgb
		
		rgbimg .: gets a gray scale( 3D matrix where each pixel is represented by 3 channels of a byte each 0..255 )

	'''	

	norm = np.linalg.norm(rgbimg)
	ch= ['r', 'g', 'b']
	channels={} 
	channels_explained_variaces={}
	channels_explained_variace_ratios={}
	# channels_singular_values= {}



	# red_img, red_explained_variance_ratio, red_singular_values = svd_compress_gs(rgbimg[:,:, 0], r=r)
	# grn_img, grn_explained_variance_ratio, grn_singular_values = svd_compress_gs(rgbimg[:,:, 1], r=r)
	# blu_img, blu_explained_variance_ratio, blu_singular_values = svd_compress_gs(rgbimg[:,:, 2], r=r)
	for i, key in enumerate(ch):
		img, explained_variance, explained_variance_ratio = svd_compress_gs(rgbimg[:,:, i], r=r)

		# print ch[i]
		# print img[0,:10]
		# print 'explained_variance:', explained_variance
		# print 'explained_variance_ratio:', explained_variance_ratio

		# channels[i] = img 
		# channels_explained_variaces[i]= explained_variance
		# channels_explained_variace_ratios[i]= explained_variance_ratio
		


if __name__ == '__main__':
	img = get_lena()
	svd_compress_rgb(img, r=3)