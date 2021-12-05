
# https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h

def convert(size, box):
    xmin, ymin, xmax, ymax = box
    dw = 1./size[0]
    dh = 1./size[1]
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x,y,w,h)
