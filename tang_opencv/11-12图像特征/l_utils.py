import matplotlib.pyplot as plt

def plt_show_bgr(imgs, titles=None, gray='', *args, **kw):
	if not isinstance(imgs, (list, tuple)):
		imgs = (imgs,)

	if len(imgs) > 2:
		row = 2
		col = (len(imgs)+1)//2
	else:
		row = 1
		col = len(imgs)
	for i, img in enumerate(imgs):
		plt.subplot(row, col, i+1)
		if isinstance(titles, (list, tuple)) and len(imgs) == len(titles):
			plt.title(titles[i])
		elif isinstance(titles, str):
			plt.title(titles)
		if gray:
			plt.imshow(img, gray)
		else:
			img = img[...,::-1]
			plt.imshow(img)
		plt.axis('off')
	plt.show()