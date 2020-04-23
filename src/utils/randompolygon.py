from PIL import Image, ImageDraw
from src.utils.randomcolor import RandomColor
import math
import random
import numpy as np


def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts ) :
    '''Start with the centre of the polygon at ctrX, ctrY, 
    then creates the polygon by sampling points on a circle around the centre. 
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    spikeyness = clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)
        points.append( (int(x),int(y)) )

        angle = angle + angleSteps[i]

    return points


def clip(x, min, max):
    if min > max:
        return x
    elif x < min:
        return min
    elif x > max:
        return max
    else:
        return x


def binRand():
    return round(random.random())


def getRandomColor():
    if binRand == 0:  # Random Primary Color
        return (binRand() * 255, binRand() * 255, binRand() * 255)
    else:  # Random Pastel Color
        return tuple(RandomColor().generate(format_='rgbArray')[0])


def randomPolygons(im_size, num_shapes, format_='im'):
    im = Image.new('RGB', (im_size, im_size), getRandomColor())
    # imPxAccess = im.load()
    draw = ImageDraw.Draw(im)

    for _ in range(num_shapes):
        verts = generatePolygon(ctrX=random.randint(0, im_size), 
                                ctrY=random.randint(0, im_size),
                                aveRadius=random.randint(1, im_size // 4), 
                                irregularity=random.choice([0.0, 0.1, 0.5]), 
                                spikeyness=random.choice([0.0, 0.1, 0.2]),
                                numVerts=random.choice([3, 4, 5, im_size // 4]))
        
        if binRand():  # Draw a filled polygon
            draw.polygon(verts, outline=None, fill=getRandomColor() )
        else:  # Draw a polyline
            draw.line(verts+[verts[0]], width=2, fill=getRandomColor() )
    
    if format_ == 'im':
        return im
    elif format_ == 'np':
        return np.array(im)


if __name__ == '__main__':
    im_size = 256
    num_shapes = random.randint(1, 6)
    
    print(randomPolygons(im_size, num_shapes, 'np'))
    randomPolygons(im_size, num_shapes, 'im').show()

# now you can save the image (im), or do whatever else you want with it.
