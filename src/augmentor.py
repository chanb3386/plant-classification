from PIL import Image, ImageEnhance
import os
import random

for f in os.listdir('flwrtrain/'):
	if f not in ".DS_Storeaugmentor.py":
		print(f)
		for i in os.listdir('flwrtrain/'+f+'/'):
			if i.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png'):
				print(i)
				img = Image.open('flwrtrain/'+f+'/'+i)

				fn, fext = os.path.splitext(i)

				print(fn+fext)

				destination = "flwrtrain/" + f + "/" + fn

				#img.convert(mode='L').save(destination + fext)
				
				#img.transpose(method=Image.FLIP_LEFT_RIGHT).save(destination + "flip" + fext)
				
				img = ImageEnhance.Brightness(img)
				img = img.enhance(random.choice([0.7,0.8,0.9,1,1.1,1.2,1.3]))
				img.save(destination + "bright" + fext)

				