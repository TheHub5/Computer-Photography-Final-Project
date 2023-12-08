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
import tkinter as tk
from tkinter import colorchooser
from PIL import ImageColor
matplotlib.use('TkAgg')

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data


def drawNewImage(canvas, image, height):
    canvas.erase()
    canvas.draw_image(data=image, location=(0, height))

# zoom in on a target location
def zoom(np_image, tarY, tarX, multi):
    # get the target dimensions
    new_w = np_image.shape[0] / multi
    new_h = np_image.shape[1] / multi

    # pixels around the target pixel
    halfSizeW = int(new_w/2)
    halfSizeH = int(new_h/2)

    new_np_image = np.zeros((int(2 * halfSizeW) +1, int(halfSizeH * 2) +1, 3)) # create new array the size of the target image

    # where to start and stop extracting pixels from the original image
    startX = tarX- halfSizeW
    startY = tarY- halfSizeH
    endX = tarX + halfSizeW
    endY = tarY + halfSizeH

    # how much to shift the image
    shiftX = 0
    shiftY = 0

    # shift the image if the target pixels end up off screen
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

    # extract
    new_np_image[startX + shiftX:endX + shiftX, startY + shiftY:endY + shiftY, :] = np_image[startX:endX, startY:endY, :]
    
    return(linInterp(new_np_image, np_image.shape[0], np_image.shape[1]))


# preform 2d linear interpolation
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

# preform 1d linear interpolation
def linInterp1D(tar, array, tarSize):
    ratio = (array.shape[0] - 1) / (tarSize - 1)

    x1 = math.floor(tar * ratio)
    x2 = math.ceil(tar * ratio)

    result = np.zeros(3)

    result[0] = (x2 - (tar * ratio)) * array[x2, 0] + ((tar * ratio) - x1) * array[x1, 0]
    result[1] = (x2 - (tar * ratio)) * array[x2, 1] + ((tar * ratio) - x1) * array[x1, 1]
    result[2] = (x2 - (tar * ratio)) * array[x2, 2] + ((tar * ratio) - x1) * array[x1, 2]

    return result

# draw a line
def drawLine(image, x1, y1, x2, y2, colour, thickness):
    xDif = x2 - x1
    yDif = y2 - y1

    m = yDif/xDif

    for z in range(xDif):
        image[(x1 + z), int(y1 + (m * z)) : int(y1 + (m * z)) + thickness, :] = colour
    
    return image

# replace the selected colour with another selected colour
def replaceColour(image, tarCol, newCol, sens):
    new_np_image = image.copy()

    cSens = ((sens / 100)/3) # distribute the sensitivity evenly and convert to decimal
    colRange = np.zeros((3, 2))
    

    if(cSens > 0):
        # if sens is greater than 0 find the range of colour to be the target
        colRange[0, 0] = tarCol[0] - int(cSens * 255)
        colRange[0, 1] = tarCol[0] + int(cSens * 255)

        colRange[1, 0] = tarCol[1] - int(cSens * 255)
        colRange[1, 1] =  tarCol[1] + int(cSens * 255)

        colRange[2, 0] = tarCol[2] - int(cSens * 255)
        colRange[2, 1] =  tarCol[2] + int(cSens * 255)

        colRange[0, 0], colRange[0, 1] = checkColourLim(colRange[0, 0], colRange[0, 1])
        colRange[1, 0], colRange[1, 1] = checkColourLim(colRange[1, 0], colRange[1, 1])
        colRange[2, 0], colRange[2, 1] = checkColourLim(colRange[2, 0], colRange[2, 1])
    else:
        # if less then 0 then only target the selected colour
        colRange[:, 0] = tarCol
        colRange[:, 1] = tarCol

    # loop through image lookign for pixels in the colour range and replace with selected colour
    for x in range(new_np_image.shape[0]):
        for y in range(new_np_image.shape[1]):
            if(checkColSim(colRange, new_np_image[x, y, :])):
                new_np_image[x, y, :] = newCol

    return(new_np_image)

def checkColSim(colRange, tarCol):
    for c in range(3):
        if(tarCol[c] < colRange[c,0] or tarCol[c] > colRange[c,1]):
            return False

    return True


def checkColourLim(min, max):
    if(max > 255):
        min = min - (max - 255)
        max = 255
        if(min < 0):
            return 0, 255
        
    elif(min < 0):
        max = max - min
        min = 0
        if(max > 255):
            return 0, 255
    
    return min, max
 
def hex_to_rgb(hex_color):
    return ImageColor.getcolor(hex_color, "RGB")

def choose_color():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    color_code = colorchooser.askcolor(title="Choose color")[1]
    root.destroy()
    return color_code

def apply_vintage_filter(image):
    #vintage filter (black and white) with film grain
    
    # Convert to float and normalize the image
    img_float = np.float32(image) / 255.0

    # Desaturate the image
    img_desaturated = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
    img_colored = cv2.cvtColor(img_desaturated, cv2.COLOR_GRAY2BGR)

    # Apply a sepia tone
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    img_sepia = cv2.transform(img_colored, sepia_filter)

    # Add film grain
    noise = np.random.randn(*img_sepia.shape) * 0.05  # Reduced the grain intensity
    img_sepia += noise

    # Apply a vignette effect
    rows, cols = img_sepia.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, 150)  # Adjusted kernel size
    kernel_y = cv2.getGaussianKernel(rows, 150)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    vignette = np.copy(img_sepia)

    for i in range(3):
        vignette[:, :, i] *= mask

    # Normalize and convert back to uint8
    vignette = np.clip(vignette, 0, 1)
    vignette = np.uint8(vignette * 255)

    return vignette

def pixelate_effect(image, kernel_size=7):
    #pixelate the image
    
    # Convert image to a suitable format for processing
    img = np.array(image, dtype=np.uint8)
    
    # Create an empty image to store the oil painting effect
    pixelate_image = np.zeros_like(img)

    # Process the image in patches
    for y in range(0, img.shape[0], kernel_size):
        for x in range(0, img.shape[1], kernel_size):
            for c in range(3):  # For each color channel
                # Extract the patch
                patch = img[y:y+kernel_size, x:x+kernel_size, c]
                
                # Find the most frequent color in the patch
                if patch.size > 0:
                    (values, counts) = np.unique(patch, return_counts=True)
                    dominant = values[np.argmax(counts)]
                    pixelate_image[y:y+kernel_size, x:x+kernel_size, c] = dominant

    return pixelate_image
    

def apply_comic_effect(image): 
    #comic book effect
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur
    blurred = cv2.medianBlur(gray, 7)
    
    # Detect edges in the image
    edges = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)
    
    # Convert back to color
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Reduce the color palette
    img_small = cv2.pyrDown(image)
    num_colors = 8
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2Lab)
    img_small = img_small.reshape((-1, 3))

    # Use OpenCV's k-means clustering
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 1.0)
    _, labels, centers = cv2.kmeans(np.float32(img_small), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reconstruct the quantized image
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape((image.shape[0] // 2, image.shape[1] // 2, 3))
    quantized = cv2.cvtColor(quantized, cv2.COLOR_Lab2BGR)
    quantized = cv2.pyrUp(quantized)
    
    # Combine edges and color-quantized image
    comic_img = cv2.bitwise_and(quantized, edges_colored)
    
    return comic_img


def display_image(np_image):

    original_image = np_image.copy()
    modified_image=original_image
    modified_image_data = original_image.copy()
    
    colour = [0,0,0] #black default colour
    replacement_color = [255, 0, 0]  # Red defualt replacement colour 
  
    # Convert numpy array to data that sg.Graph can understand
    image_data = np_im_to_data(np_image)

    height = np_image.shape[0]
    width = np_image.shape[1]

    hist = construct_image_histogram(np_image)
    fig = plt.figure(figsize=(5,4),dpi=100)
    fig.add_subplot(111).bar(np.arange(len(hist)), hist)
    plt.title('Histogram')

    colour = np.zeros(3)

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
        [[sg.Column([[sg.Text('Sensitivity:'), sg.Slider((0, 100), 15, 1, orientation='horizontal', key='-sens-')]], justification='center')]],
        [[sg.Column([[sg.Button('Check Location')]], justification='center')]],
        

        [sg.Button('Exit'),
        sg.Button('Save', button_color=('orange')),
        sg.Push(),
        
        sg.Button('Comic Book Effect', button_color=('green')),
        sg.Button('Pixelate Effect',button_color=('green')),
        sg.Button('Vintage Effect',button_color=('green')),
        sg.Button('Replace Colour'),
        sg.Button('Select Colour'),
        sg.Button('Reset',button_color=('red')),
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

            # check if valid coords
            if(x > (np_image.shape[1] - 1) or y > (np_image.shape[0] - 1) or x < 0 or y < 0):
                sg.popup_no_buttons('Selected out of bounds x/y', keep_on_top=True)
            else:
                colour = np_image[y, x, :] # extract colour from selected pixel
                new_np_image = np_image

                new_np_image = zoom(new_np_image, x, y, 2) # create zoomed in image

                # create border
                new_np_image[0:boxSize, 0:boxSize, :] = 0
                new_np_image[0:boxSize, 0:boxSize, 0] = 255


                # fill with selected colour
                new_np_image[boxWidth:boxSize - boxWidth, boxWidth:boxSize - boxWidth, 0] = colour[0]
                new_np_image[boxWidth:boxSize - boxWidth, boxWidth:boxSize - boxWidth, 1] = colour[1]
                new_np_image[boxWidth:boxSize - boxWidth, boxWidth:boxSize - boxWidth, 2] = colour[2]

                # draw a line from the box to the selected pixel
                new_np_image = drawLine(new_np_image, boxSize - boxWidth, boxSize - boxWidth, int(new_np_image.shape[0]/ 2), int(new_np_image.shape[1]/ 2), np.array([255, 0, 0]), 3)

                # dispaly the new image
                new_image_data = np_im_to_data(new_np_image)
                drawNewImage(window['-IMAGE2-'], new_image_data, height)

        elif event == 'Select Colour':
            # Open color chooser dialog
            chosen_color = choose_color()
            if chosen_color:
                replacement_color = hex_to_rgb(chosen_color)
        
        elif event == 'Replace Colour':
            # Replace the selected color in the image
                    if replacement_color:
                        print ("Color Replaced!")
                        new_np_image = np_image
                        new_np_image = replaceColour(np_image, colour, replacement_color,values['-sens-'])
                        modified_image = new_np_image #update modified_image
                        new_image_data = np_im_to_data(new_np_image)
                        drawNewImage(window['-IMAGE2-'], new_image_data, height)
                        

                
        elif event == 'Vintage Effect':
            vintage_image = apply_vintage_filter(np_image)
            modified_image = vintage_image #update modified_image
            vintage_image_data = np_im_to_data(vintage_image)
            drawNewImage(window['-IMAGE2-'], vintage_image_data, vintage_image.shape[0])  # Replace '-IMAGE-' with your actual image display element key

        
        elif event == 'Pixelate Effect':
            
            pixelate_image = pixelate_effect(np_image)
            modified_image = pixelate_image #update modified_image
            pixelate_image_data = np_im_to_data(pixelate_image)
            drawNewImage(window['-IMAGE2-'], pixelate_image_data, pixelate_image.shape[0])
            
        elif event == 'Comic Book Effect':
            # Apply the comic book effect to the image
            comic_image = apply_comic_effect(np_image)
            modified_image = comic_image #update modified_image
            # Update the GUI to display the comic book image
            # You might need to convert the image to a format suitable for your GUI
            comic_image_data = np_im_to_data(comic_image)
            drawNewImage(window['-IMAGE2-'], comic_image_data, comic_image.shape[0])

        elif event == 'Reset':
            image_data = np_im_to_data(np_image)
            drawNewImage(window['-IMAGE2-'], image_data, height)
            

        elif event == 'Save':
            if modified_image is not None:
                # Ensure the image is in the correct format
                if modified_image.dtype != np.uint8:
                    modified_image = np.clip(modified_image, 0, 255).astype(np.uint8)
                
                # Convert the image to BGR format for OpenCV
                image_to_save = cv2.cvtColor(modified_image, cv2.COLOR_RGB2BGR)
                
                # Open a 'Save As' dialog to get the filename from the user
                save_filename = sg.popup_get_file('Save Image As', save_as=True, no_window=True, file_types=(('PNG Files', '*.png'), ('JPEG Files', '*.jpg'), ('All Files', '*.*')), default_extension='png')
                
                if save_filename and save_filename.strip():
                    # Ensure the filename has an extension
                    if not any(save_filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
                        sg.popup('Error', 'Please provide a valid image file extension for the file.')
                    else:
                        # Save the image using the filename provided by the user
                        try:
                            cv2.imwrite(save_filename, image_to_save)
                            sg.popup('Image saved!', f'File has been saved to: {save_filename}')
                        except Exception as e:
                            sg.popup('Error', f'Failed to save the image: {e}')
                else:
                    sg.popup('Save Cancelled', 'Image save operation was cancelled.')    

            


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