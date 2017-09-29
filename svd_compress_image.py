'''
	Created on Sep 29, 2017

	@author: Varela

	ref: Blum. Foundations of Data Science Exercise 3.27, using TruncatedSVD
			We 

'''

import numpy as np 
import matplotlib.pyplot as plt 
from utils import get_lena


def svd_compress_gs(gsimg, r=3, verbose=True):
	'''
		Applies SVD to a gray scale matrix

		gsimg .: gets a gray scale( 2D matrix where each pixel is represented by a 1 byte 0..255 ) and applies truncated SVD

	'''	

	u, s, v = np.linalg.svd(gsimg, full_matrices=True)

	if verbose: 
		# print 'shapes: U:',u.shape,'\tS: ',s.shape,'\tV:\t',v.shape
		print 'explained_variance:', np.cumsum(s[:r]) / s.sum() 
		print 'total_explained_variance:', s[:r].sum() / s.sum() 
	
	S = np.diag(s[:r])	
	U = u[:,:r]
	V = v[:r,:]
	
	newimg = U.dot(S.dot(V)).astype(np.uint8)

	return newimg, u, s, v


def svd_compress_rgb(rgbimg, r=3, verbose=True):
	'''
		reuses svd_compress_rgb
		
		rgbimg .: gets a gray scale( 3D matrix where each pixel is represented by 3 channels of a byte each 0..255 )

	'''	

	norm = np.linalg.norm(rgbimg)
	ch= ['red', 'green', 'blue']
	channels={} 
	channels_u={}
	channels_s={}
	channels_v={}


	stats_explained_variance={}
	stats_explained_variance_ratio={}
	
	explained_variances= [] 
	total_variances=	[]
	for i, key in enumerate(ch):	


		newimg, u, s, v = svd_compress_gs(rgbimg[:,:, i], r=r)	

		channels[i] = newimg 
		channels_u[i]= u
		channels_s[i]= s
		channels_v[i]= v

		stats_explained_variance[i] = np.cumsum(s[:r]) 
		stats_explained_variance_ratio[i] = np.cumsum(s[:r]) / np.sum(s)

		explained_variances.append(stats_explained_variance[i][-1])
		total_variances.append(np.sum(s))
		
	


	newimg = np.dstack(tuple(channels.values()))

	ratio = sum(explained_variances) / sum(total_variances)

	if verbose:
		plt.title('Reconstructed image (explained=%.2f)' % (ratio))
		plt.imshow(newimg)
		# plt.show()
	
	return newimg, ratio 		

def main():
	img = get_lena()

	imgnorm = np.linalg.norm(img)
	min_size = min(img.shape[0:2])
	R = [0.05, 0.10 , 0.25, 0.50]
	dims= [] 
	images=[] 
	ratios=[] 
	norms= [] 
	for r in R:
		d = int(min_size*r)
		newimg, ratio= svd_compress_rgb(img, r=d, verbose=False)

		norm = np.linalg.norm(newimg) / imgnorm
		dims.append(d)
		images.append(newimg)
		ratios.append(ratio)
		norms.append(norm) 
	# Four axes, returned as a 2-d array
	f, axarr = plt.subplots(2, 2, figsize=(12,12))

	axarr[0, 0].imshow(images[0])
	axarr[0, 0].set_title('dims=%d, ratio=%.2f, forbenius=%.2f' % (dims[0], ratios[0], norms[0]), fontsize=10 )
	axarr[0, 1].imshow(images[1])
	axarr[0, 1].set_title('dims=%d, ratio=%.2f, forbenius=%.2f' % (dims[1], ratios[1], norms[1]), fontsize=10 )
	axarr[1, 0].imshow(images[2])
	axarr[1, 0].set_title('dims=%d, ratio=%.2f, forbenius=%.2f' % (dims[2], ratios[2], norms[2]), fontsize=10 )
	axarr[1, 1].imshow(images[3])
	axarr[1, 1].set_title('dims=%d, ratio=%.2f, forbenius=%.2f' % (dims[3], ratios[3], norms[3]), fontsize=10 )
	
	# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
	plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
	plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
	plt.title('Image using various compression rates')
	plt.show()

	
if __name__ == '__main__':
	main()

