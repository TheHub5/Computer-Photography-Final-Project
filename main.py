# Faisal Z. Qureshi
# www.vclab.ca


import argparse
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
import time
import scipy as sp
from scipy import signal
import math
matplotlib.use('TkAgg')

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def construct_image_histogram(np_image):
    L = 256
    bins = np.arange(L+1)
    hist, _ = np.histogram(np_image, bins)
    return hist

def draw_hist(canvas, figure):
    tkcanvas = FigureCanvasTkAgg(figure, canvas)
    tkcanvas.get_tk_widget()
    tkcanvas.draw()
    tkcanvas.get_tk_widget().pack(side='top', fill='both', expand=1)
    return tkcanvas

def drawNewImage(canvas, image, height):
    canvas.erase()
    canvas.draw_image(data=image, location=(0, height))


def apply_filter_to_patch(patch, filter):
    return np.sum(filter * patch)

def apply_filter_to_image(image, filter):
    width = int((len(filter)-1)/2)
    result = np.ones( (len(image)- (len(filter)-1), len(image[0])- (len(filter[0])-1), 3) )
    final = image
    
    for i in range(len(result)):
        for j in range(len(result[0])):
            for c in range(3):
                # target coords
                centerX = int(j + ((len(filter)-1)/2))
                centerY = int(i + ((len(filter)-1)/2))
                result[i][j][c] = apply_filter_to_patch(image[centerY-width:centerY+width+1, centerX-width:centerX+width+1, c], filter)
    
    return result

def apply_filter_to_imagec(image, filter):
    width = int((len(filter)-1)/2)
    result = np.ones( (len(image)- (len(filter)-1), len(image[0])- (len(filter[0])-1), 3) )
    final = image
    

    for c in range(3):
        # target coords
        centerX = int(j + ((len(filter)-1)/2))
        centerY = int(i + ((len(filter)-1)/2))
        result[:][:][c] = sp.signal.convolve2d(np_image[:, :, c], filter, mode='same')
    
    return result

def createAvgKernel(halfSize):
    size = 2*halfSize+1
    f = (np.ones((size, size)))
    f = f/np.sum(f)
    return f

def createGausBlur(halfSize, mu, sigma):
    size = 2*halfSize+1

    gy_ = np.empty((size))

    for i in range(-halfSize,halfSize):
        gy_[i + halfSize] = g(mu, sigma, i)

    
    gx_ = gy_.reshape((size, 1))
    g_ = gx_ * gy_

    g_ = g_ / np.sum(g_)
    return g_

def g(mu, s, x):
    exponent = -((x - mu)**2)/(2*s*s)
    constant = 1/(s * np.sqrt(2 * np.pi))
    return constant * np.exp( exponent )

def saveSettings(sat, con, pal, path):
    settings = {'sat': sat,'con': con,'pal': pal}
    with open(file_path, 'w') as file:
        yaml.dump(settings, file, default_flow_style=False)

def saturation(image, sat):
    himage = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    himage[:, :, 1] = himage[:, :, 1] * sat
    himage[:, :, 1] = np.clip(himage[:,:, 1], 0, 255)
    return cv2.cvtColor(himage, cv2.COLOR_HSV2RGB)

def contrast(image, con):
    image = np.float32(image)/255.0
    con = np.clip(con, 0.5, 2.5)
    conscale = 2 * (con - 0.5)

    
    if conscale >= 0:
        a = 10.0 * conscale 
    else:
        a = -10.0/(1 + np.exp(-conscale)) 

    v = 1 + np.exp(-a*(image - 0.5))
    curve = 1/v

    if con == 1.0:
        curve = image

    cimage = np.uint8(curve*255)

    return cimage

def palette(image, pal):
    image = np.float32(image)/255.0

    if pal > 0:
        image[:, :, 2] = image[:, :, 2] + pal / 100
        image[:, :, 2] = np.clip(image[:, :, 2], 0, 1)
    elif pal < 0:
        image[:, :, 0] = image[:, :, 0] - pal / 100
        image[:, :, 0] = np.clip(image[:, :, 0], 0, 1)

    return np.uint8(image * 255)

# zoom in on a target location
def zoom(np_image, tarY, tarX, multi):
    new_w = np_image.shape[0] / multi
    new_h = np_image.shape[1] / multi

    print("test", new_w)

    halfSizeW = int(new_w/2)
    halfSizeH = int(new_h/2)



    print(halfSizeW)

    new_np_image = np.zeros((int(2 * halfSizeW) +1, int(halfSizeH * 2) +1, 3))

    startX = tarX- halfSizeW
    startY = tarY- halfSizeH
    endX = tarX + halfSizeW
    endY = tarY + halfSizeH

    shiftX = 0
    shiftY = 0

    shiftX2 = 0
    shiftX2 = 0

    if((tarX- halfSizeW) < 0):
        shiftX = -(tarX- halfSizeW)
        startX = 0
    if(tarX+ halfSizeW > new_np_image.shape[0]):
        shiftX = -startX
    if((tarY- halfSizeH) < 0):
        shiftY = -(tarY- halfSizeH)
        startY = 0
    if(tarY+ halfSizeH > new_np_image.shape[1]):
        shiftY = -startY

    if(tarX+ halfSizeW > np_image.shape[0]):
        endX = np_image.shape[0]
        shiftX = -startX
    if(tarY+ halfSizeH > np_image.shape[1]):
        endY = np_image.shape[1]
        shiftY = -startY
    new_np_image[startX + shiftX:endX + shiftX, startY + shiftY:endY + shiftY, :] = np_image[startX:endX, startY:endY, :]
    
    
    print(tarX- int(new_w/2))
    return(linInterp(new_np_image, np_image.shape[0], np_image.shape[1]))
    # return(new_np_image)


def linInterp(np_image, tarW, tarH):
    ratioW = (np_image.shape[0] - 1) / (tarW - 1)
    ratioH = (np_image.shape[1] - 1) / (tarH - 1)
    
    result = np.zeros((tarW, tarH, 3))

    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            y1 = math.floor(y * ratioH)
            y2 = math.ceil(y * ratioH)

            a1 = linInterp1D(x, np_image[:, y1, :], tarW)
            a2 = linInterp1D(x, np_image[:, y2, :], tarW)
            
            result[x, y, 0] = (y2 - (y * ratioH)) * a2[0] + ((y * ratioH) - y1) * a1[0]
            result[x, y, 1] = (y2 - (y * ratioH)) * a2[1] + ((y * ratioH) - y1) * a1[1]
            result[x, y, 2] = (y2 - (y * ratioH)) * a2[2] + ((y * ratioH) - y1) * a1[2]   

    return result

def linInterp1D(tar, array, tarSize):
    ratio = (array.shape[0] - 1) / (tarSize - 1)

    x1 = math.floor(tar * ratio)
    x2 = math.ceil(tar * ratio)

    result = np.zeros(3)

    result[0] = (x2 - (tar * ratio)) * array[x2, 0] + ((tar * ratio) - x1) * array[x1, 0]
    result[1] = (x2 - (tar * ratio)) * array[x2, 1] + ((tar * ratio) - x1) * array[x1, 1]
    result[2] = (x2 - (tar * ratio)) * array[x2, 2] + ((tar * ratio) - x1) * array[x1, 2]

    return result




def display_image(np_image):
    
    # Convert numpy array to data that sg.Graph can understand
    image_data = np_im_to_data(np_image)

    height = np_image.shape[0]
    width = np_image.shape[1]

    hist = construct_image_histogram(np_image)
    fig = plt.figure(figsize=(5,4),dpi=100)
    fig.add_subplot(111).bar(np.arange(len(hist)), hist)
    plt.title('Histogram')

    # Define the layout
    layout = [[sg.Graph(
        canvas_size=(width, height),
        graph_bottom_left=(0, 0),
        graph_top_right=(width, height),
        key='-IMAGE-',
        background_color='white',
        change_submits=True,
        drag_submits=True),
        sg.Graph(
        canvas_size=(width, height),
        graph_bottom_left=(0, 0),
        graph_top_right=(width, height),
        key='-IMAGE2-',
        background_color='white',
        change_submits=True,
        drag_submits=True)],
        [[sg.Column([[sg.Text('x:'), sg.Input('', key='-X-')]], justification='center')]],
        [[sg.Column([[sg.Text('y:'), sg.Input('', key='-Y-')]], justification='center')]],
        [[sg.Column([[sg.Button('Check Location')]], justification='center')]],

        [sg.Button('Exit'),
        sg.Push(),
        sg.Button('Replace Colour'),
        ]]
    # gaussian
    # Create the window
    window = sg.Window('Display Image', layout, finalize=True)    
    window['-IMAGE-'].draw_image(data=image_data, location=(0, height))


    # Event loop
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        elif event == 'Check Location':
            boxSize = 50
            boxWidth = 5

            x = int(values['-X-'])
            y = int(values['-Y-'])
            colour = np_image[y, x, 0:3]

            new_np_image = zoom(np_image, x, y, 2)

            # create border
            new_np_image[0:boxSize, 0:boxSize, :] = 0
            new_np_image[0:boxSize, 0:boxSize, 0] = 255


            # fill with selected colour
            new_np_image[boxWidth:boxSize - boxWidth, boxWidth:boxSize - boxWidth, 0] = colour[0]
            new_np_image[boxWidth:boxSize - boxWidth, boxWidth:boxSize - boxWidth, 1] = colour[1]
            new_np_image[boxWidth:boxSize - boxWidth, boxWidth:boxSize - boxWidth, 2] = colour[2]


            new_image_data = np_im_to_data(new_np_image)

            drawNewImage(window['-IMAGE2-'], new_image_data, height)

            


    window.close()

def main():
    parser = argparse.ArgumentParser(description='A simple image viewer.')

    parser.add_argument('file', action='store', help='Image file.')
    args = parser.parse_args()

    print(f'Loading {args.file}... ', end='\n')
    image = cv2.imread(args.file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'Image shape is {image.shape}... ', end='\n')
 

    # check if horizontal or verticle image apply approriate scaling to avoid cropping
    if(image.shape[0] < image.shape[1]):
        print(f'Resizing the image to {int(np.rint((image.shape[0] * (640 / image.shape[1]))))}x640 a...', end='\n')
        print(f'Scaling by {640 / image.shape[1]}...' , end='\n')
        image = cv2.resize(image, (640, int(np.rint((image.shape[0] * (640 / image.shape[1]))))), interpolation=cv2.INTER_LINEAR)
    else:
        print(f'Resizing the image to 480x{int(np.rint((image.shape[1] * (480 / image.shape[0]))))} ...', end='\n')
        print(f'Scaling by {480 / image.shape[0]}...' , end='\n')
        image = cv2.resize(image, (int(np.rint((image.shape[1] * (480 / image.shape[0])))), 480), interpolation=cv2.INTER_LINEAR)
    print(f'Resized image shape is {image.shape}...', end='\n')

    display_image(image)

if __name__ == '__main__':
    main()