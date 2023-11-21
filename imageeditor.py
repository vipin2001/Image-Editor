from tkinter import *
from PIL import ImageTk,Image 
from tkinter import filedialog
from tkinter import simpledialog
import tkinter.font as font
import numpy as np
from numpy import asarray
import cv2
import math

#To know if we are working on grayscale Image or Color Image
global grayimg
grayimg = False

#Global Variable Defination created in function
#img list : Contains a list of matrix of image after applying new feature to the image
#Currimage : The current Image being currently dispaled on canvas (in ImageTk format)
#currimagematrix : The current Image being currently dispaled on canvas (in RGB array format)

#To display on canvas after processing the image
def showimage():
	global currimage
	canvas.itemconfig(currimg_container,image=currimage)
	canvas.configure(scrollregion=canvas.bbox("all"))

#Converting grayscale image to RGB
def grayscale_rgb():
	
	global currimage
	global currimagematrix
	grayimg = True
	#To make the grayscale image to RGB by making R=G=B= Grayscale(value)

	new_matrix = np.zeros((currimagematrix.shape[0],currimagematrix.shape[1],3))
	max_intensity = np.max(currimagematrix)
	min_intensity = np.min(currimagematrix)
	extent = max_intensity - min_intensity
	
	currimagematrix = (currimagematrix-min_intensity)*255.0/extent
	#new_matrix[:,:,0] = currimagematrix
	#new_matrix[:,:,1] = currimagematrix
	new_matrix[:,:,2] = currimagematrix
	new_matrix =cv2.cvtColor(new_matrix.astype(np.uint8), cv2.COLOR_HSV2BGR)
	currimagematrix = new_matrix.astype(np.uint8)

#Loading an Image onto a canvas
def loadImage():
	global imglist
	#Clearing the Previous Image stack
	imglist = []
	global currimage
	global currimagematrix
	#Asking the user for the filepath of the image
	filepath = filedialog.askopenfilename()
	cimage = Image.open(filepath)
	currimage = ImageTk.PhotoImage(cimage)
	#Obtaining the array of the image
	currimagematrix = np.array(cimage)
	if len(currimagematrix.shape) == 2:
		grayscale_rgb()
	#Adding the Image to stack for UNDO if required
	imglist.append(currimagematrix)
	currimage = ImageTk.PhotoImage(Image.fromarray(currimagematrix))
	showimage()

#Saving an Image
def saveImage():
	global currimagematrix
	#Asking the user for the filepath and file nameof the image
	filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
	#Converting to image format from array
	image_final = Image.fromarray(currimagematrix)
	image_final.save(filename)

#Undo an Image
def undoImage():
	global imglist
	global currimage
	global currimagematrix
	if len(imglist)<=1:
		return
	#Deleting the latest entry to the stack
	temp = imglist.pop(len(imglist)-1)
	#Updating the current image to the last entry of the Updated Stack
	currimagematrix = imglist[(len(imglist)-1)]
	currimage = ImageTk.PhotoImage(Image.fromarray(currimagematrix))
	showimage()

#Load the original image
def orgImage():
	global imglist
	global currimage
	global currimagematrix
	#Obtaining the Image loaded, i.e. the first image pushed to the stack
	currimagematrix = imglist[0]
	#clearing the stack
	imglist = []
	currimage = ImageTk.PhotoImage(Image.fromarray(currimagematrix))
	#Updating the stack
	imglist.append(currimagematrix)
	showimage()
#Apply histogram equalisation to the image
def histeqImage():
	global imglist
	global currimage
	global currimagematrix
	#Converting RGB to HSV
	hsvImage = cv2.cvtColor(currimagematrix, cv2.COLOR_BGR2HSV)
	#Creating temprory array for Histogram formation
	d = hsvImage[:,:,2]

	#Creating ping of size 1
	mybin = [i for i in range(257)]
	#craeting histogram
	hist = np.histogram(d,bins=mybin)[0]
	#Normalized Histogram
	hist = hist/(hsvImage.shape[0]*hsvImage.shape[1])
	#Cummulative Sum of the histogram
	hist = np.cumsum(hist)
	# Scaling upto the Intensity Level
	hist = hist*(255)
	#Rounding the value
	hist = np.rint(hist)
	hsvImage[:,:,2] = hist[hsvImage[:,:,2]]

	rgbImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
	currimagematrix = rgbImage
	imglist.append(rgbImage)
	currimage = ImageTk.PhotoImage(Image.fromarray(rgbImage))
	showimage()
#Histogram Matching
def histMatching(target_cdf):
    global imglist
    global currimage
    global currimagematrix
    
    # Converting RGB to HSV
    hsvImage = cv2.cvtColor(currimagematrix, cv2.COLOR_BGR2HSV)
    
    # Extract the intensity channel (Value component in HSV)
    intensity_channel = hsvImage[:,:,2]

    # Calculate the histogram of the intensity channel
    hist, bins = np.histogram(intensity_channel.flatten(), bins=256, range=[0, 256])

    # Normalize the histogram
    hist_normalized = hist / (intensity_channel.shape[0] * intensity_channel.shape[1])

    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(hist_normalized)

    # Map pixel values based on the CDF for histogram equalization
    intensity_equalized = np.interp(intensity_channel.flatten(), bins[:-1], cdf * 255)
    intensity_equalized = np.reshape(intensity_equalized, intensity_channel.shape).astype(np.uint8)

    # Apply histogram equalization to the Value channel
    hsvImage[:,:,2] = intensity_equalized

    # Perform histogram specification 
    intensity_specified = np.interp(intensity_channel.flatten(), cdf * 255, target_cdf)
    intensity_specified = np.reshape(intensity_specified, intensity_channel.shape).astype(np.uint8)

    # Apply histogram specification to the Value channel
    hsvImage[:,:,2] = intensity_specified

    # Convert the result back to BGR
    rgbImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
    
    # Update the current image matrix
    currimagematrix = rgbImage
    
    # Append the image to the list
    imglist.append(rgbImage)
    
    # Convert the image to PhotoImage for display
    currimage = ImageTk.PhotoImage(Image.fromarray(rgbImage))
    
    # Show the image
    showimage()


#Apply Gamma correction to the Image
def gcorrectImage():
	global imglist
	global currimage
	global currimagematrix
	#Requesting User for gamma Value
	gamma = float(simpledialog.askstring(title="Gamma",prompt="Please enter gamma"))
	#Obtaining HSV image from RGB
	hsvImage = cv2.cvtColor(currimagematrix, cv2.COLOR_BGR2HSV)
	#Applying Gamma Correction
	hsvImage[:,:,2] = pow(1/255,gamma-1)*pow(hsvImage[:,:,2],gamma)
	rgbImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
	currimagematrix = rgbImage
	imglist.append(rgbImage)
	currimage = ImageTk.PhotoImage(Image.fromarray(rgbImage))
	showimage()

#Apply log transform to the image
def ltransImage():
	global imglist
	global currimage
	global currimagematrix
	#Obtaining HSV image from RGB
	hsvImage = cv2.cvtColor(currimagematrix, cv2.COLOR_BGR2HSV)
	#Appling Log transform
	hsvImage[:,:,2] = (255.0/math.log(256))*np.log(1+hsvImage[:,:,2])
	rgbImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
	currimagematrix = rgbImage
	imglist.append(rgbImage)
	currimage = ImageTk.PhotoImage(Image.fromarray(rgbImage))
	showimage()

#Blurring the Image
def blurImage():
	global imglist
	global currimage
	global currimagematrix
	#Asking user for blur extent
	blur_extent = int(simpledialog.askstring(title="Blur",prompt="Please enter blur extent"))
	#Making the kernel for convolution
	kernel = (np.zeros(((2*blur_extent-1),(2*blur_extent-1))) + 1)/((2*blur_extent-1)*(2*blur_extent-1))
	hsvImage = cv2.cvtColor(currimagematrix, cv2.COLOR_BGR2HSV)
	#Convolution for blue
	hsvImage = conv_filter(hsvImage,kernel)
	rgbImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
	currimagematrix = rgbImage
	imglist.append(rgbImage)
	currimage = ImageTk.PhotoImage(Image.fromarray(rgbImage))
	showimage()

#Convolution Filer to the HSV matrix
def conv_filter(main_matrix,kernel_matrix):
	#Padding the Image
	padded_image = np.zeros((main_matrix.shape[0]+(2*(kernel_matrix.shape[0]//2)),(main_matrix.shape[1]+(2*(kernel_matrix.shape[1]//2))),3))
	padded_image[(kernel_matrix.shape[0]//2):(-1*(kernel_matrix.shape[0]//2)),(kernel_matrix.shape[1]//2):-(kernel_matrix.shape[1]//2),2] = main_matrix[:,:,2] 
	#Matrix to contain the update value
	new_matrix = np.zeros(main_matrix.shape)
	for i in range(main_matrix.shape[0]):
		i = i + kernel_matrix.shape[0]//2
		for j in range(main_matrix.shape[1]):
			j+=kernel_matrix.shape[1]//2
			#Convolution
			main_matrix[i-kernel_matrix.shape[0]//2][j-kernel_matrix.shape[1]//2][2] = np.sum(np.multiply(kernel_matrix,padded_image[(i-(kernel_matrix.shape[0]//2)):(i+(kernel_matrix.shape[0]//2)+1),(j-(kernel_matrix.shape[1]//2)):(j+(kernel_matrix.shape[1]//2+1)),2]))
	return main_matrix
def gaussianfilter(hsv_matrix):
	kernel = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273
	gaussian = conv_filter(hsv_matrix,kernel)
	return gaussian

#Sharpening the Image
def sharpImage():
	global imglist
	global currimage
	global currimagematrix
	#Obtaining HSV image from RGB
	hsvImage = cv2.cvtColor(currimagematrix, cv2.COLOR_BGR2HSV)
	#Asking user for Sharp extent
	sharp_extent = float(simpledialog.askstring(title="Sharp",prompt="Please enter Sharp extent"))
	#Gausian Filter Unmasking
	gaussian = gaussianfilter(hsvImage)
	gaussian[:,:,2] = cv2.cvtColor(currimagematrix, cv2.COLOR_BGR2HSV)[:,:,2] - gaussian[:,:,2]
	hsvImage[:,:,2] = cv2.cvtColor(currimagematrix, cv2.COLOR_BGR2HSV)[:,:,2] + sharp_extent/10*gaussian[:,:,2]
	max_intensity = np.max(hsvImage[:,:,2])
	min_intensity = np.min(hsvImage[:,:,2])
	extent = max_intensity - min_intensity
	
	hsvImage[:,:,2] = (hsvImage[:,:,2]-min_intensity)*255.0/extent

    
	#np.rint(sharp_extent*lap_filter[:,:,2]).astype('uint8')

	rgbImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
	currimagematrix = rgbImage
	imglist.append(rgbImage)
	currimage = ImageTk.PhotoImage(Image.fromarray(rgbImage))
	showimage()

	
#Inverting the Intensity of Image
def colorinversionImage():
	global imglist
	global currimage
	global currimagematrix
	#Obtaining HSV image from RGB
	hsvImage = cv2.cvtColor(currimagematrix, cv2.COLOR_BGR2HSV)
	#Inverting the Intensity
	hsvImage[:,:,2] = 255 - hsvImage[:,:,2]
	rgbImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)
	currimagematrix = rgbImage
	imglist.append(rgbImage)
	currimage = ImageTk.PhotoImage(Image.fromarray(rgbImage))
	showimage()

#Defining the Parent for Tkinter (root is the parent)
root = Tk()
#Giving Title to my Image Editor
root.title("Image Editor")
#Initialize Icon for the Image Editor
icon_photo = PhotoImage(file = "logo.png")
root.iconphoto(False, icon_photo)
#Setting the background of the GUI
root.configure(background='black')
#The application is not resizeable
root.resizable(False, False)

#Creating a Frame to hold my Canvas and scroll bar
frame=Frame(root,width=800,height=700)
frame.grid(row=0,column=0,rowspan = 110)
#Creating Canvas
canvas = Canvas(frame, bg='black', width=800, height=700)
canvas.grid(row=0, column=0)
#Creating Scroll Bar
scroll_x = Scrollbar(frame, orient="horizontal", command=canvas.xview)
scroll_x.grid(row=1, column=0, sticky="ew")
scroll_y = Scrollbar(frame, orient="vertical", command=canvas.yview)
scroll_y.grid(row=0, column=1, sticky="ns")
canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
#Update the Scroll bar for the image
canvas.configure(scrollregion=canvas.bbox("all"))

#Loading a default Image asking the user to load a Image
currimage = ImageTk.PhotoImage(Image.open("start.png"))
currimg_container = canvas.create_image(400, 350,anchor = CENTER, image= currimage)

#Button Attributes
button_bg = '#f1592a'
button_width = 20
button_text_color = '#000000'
padx_button = 10
button_bdcolor = '#f1592a'
button_border = 5
buttonFont = font.Font(family='Helvetica', size=12	, weight='bold')

#Text Displayed to give Some information
t = Text(root,height = 5,width = 25, bg = "black",fg = "#17b9cf" ,font = buttonFont,borderwidth=0)
t.grid(row = 0, column =1, columnspan = 40,pady =5)
t.tag_configure("center", justify='center')
t.insert(END,"Vipin Singh\n19D070069\nEE610: Image Processing")
t.tag_add("center", "1.0", "end")

#Displaying Buttons for Different Feature
x = 50

#Load Image Button
load_button = Button(root, text= "LOAD", command= loadImage ,font=buttonFont, bg = button_bg, fg = button_text_color,width = button_width,borderwidth=button_border)
load_button.grid(column=1,row=x, pady = 2, padx = padx_button)

x+=1

#Save Image Button
save_button = Button(root,text="SAVE",command= saveImage,font=buttonFont, bg = button_bg, fg = button_text_color,width = button_width,borderwidth=button_border)
save_button.grid(column=1,row=x,pady = 2, padx = padx_button)

x+=1

#Undo Image Button
undo_button = Button(root,text="UNDO",command= undoImage,font=buttonFont, bg = button_bg ,fg = button_text_color,width = button_width,borderwidth=button_border)
undo_button.grid(column = 1, row = x,pady = 2, padx = padx_button)

x+=1

#Original Image Button
original_button = Button(root,text="ORIGINAL",font=buttonFont, command = orgImage, bg = button_bg, fg = button_text_color,width = button_width,borderwidth=button_border)
original_button.grid(column = 1, row = x,pady = 2, padx = padx_button)

x+=1
#Histogram Equalization Button
histeq_button = Button(root,text="Histogram Equalization", command = histeqImage, font=buttonFont,bg = button_bg, fg = button_text_color,width = button_width,borderwidth=button_border )
histeq_button.grid(column = 1, row = x,pady = 2, padx = padx_button)

x+=1

#Gamma Correcction Button
gcorrect_button = Button(root,text="Gamma Correction", command = gcorrectImage, font=buttonFont,bg = button_bg, fg = button_text_color,width = button_width,borderwidth=button_border)
gcorrect_button.grid(column = 1, row = x,pady = 2, padx = padx_button)

x+=1

#Log Transform Button
ltrans_button = Button(root,text="Log Transform", command = ltransImage,font=buttonFont, bg = button_bg, fg = button_text_color,width = button_width, borderwidth=button_border)
ltrans_button.grid(column = 1, row = x,pady = 2, padx = padx_button)

x+=1

#Blur Image Button
blur_button = Button(root,text="Blur Image", command = blurImage,font=buttonFont, bg = button_bg, fg = button_text_color,width = button_width, borderwidth=button_border)
blur_button.grid(column = 1, row = x,pady = 2, padx = padx_button)

x+=1

#Sharp Image Button
sharp_button = Button(root,text="Sharp Image", command = sharpImage, font=buttonFont,bg = button_bg, fg = button_text_color,width = button_width, borderwidth=button_border)
sharp_button.grid(column = 1, row = x,pady = 2, padx = padx_button/2)

x+=1

#Color Inversion Button
cinv_button = Button(root,text="Invert Colors", command = colorinversionImage, font=buttonFont,bg = button_bg, fg = button_text_color,width = button_width,borderwidth=button_border)
cinv_button.grid(column = 1, row = x,pady = 2, padx = padx_button)
cinv_button.config(highlightthickness=2, highlightbackground="red")

#Logo of Image Editor
logoimg = ImageTk.PhotoImage(Image.open("logo.png"))
panel = Label(root, image = logoimg,borderwidth=0,pady = 4,bg = 'black')

panel.grid(column = 1, row =x+1, rowspan =42)

root.mainloop()
