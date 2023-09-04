from PIL import Image
 
# Generating test square images...

im_size = 6

data = list()
for x in range(im_size):
    for y in range(im_size):
        if x % 2 == 0:
            color = (255, 0, 0)
        elif x % 3 == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        data.append(color)

img = Image.new('RGB', (im_size, im_size))
img.putdata(data)
img.save('test.png')
