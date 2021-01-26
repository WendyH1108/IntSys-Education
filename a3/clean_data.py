import pickle
import numpy as np
from PIL import Image, ExifTags,ImageOps



def load_pickle_file(path_to_file):
    """
    Loads the data from a pickle file and returns that object
    """
    ## Look up: https://docs.python.org/3/library/pickle.html
    ## The code should look something like this:
    # with open(path_to_file, 'rb') as f:
    #   obj = pickle....
    ## We will let you figure out which pickle operation to use
    with open(path_to_file,'rb')as f:
        new_data=pickle.load(f)
    return new_data


## You should define functions to resize, rotate and crop images
## below. You can perform these operations either on numpy arrays
## or on PIL images (read docs: https://pillow.readthedocs.io/en/stable/reference/Image.html)
        
def resize(image, height, width):
    newSize = (width, height)
    image = image.resize(newSize)
    return image
    
def crop(image, left, top, right, bottom):
    image = image.crop((left, top, right, bottom))
    return image

## We want you to clean the data, and then create a train and val folder inside
## the data folder (so your data folder in a3/ should look like: )
# data/
#   train/
#   val/

## Inside the train and val folders, you will have to dump the CLEANED images and
## labels. You can dump images/annotations in a pickle file (because our data loader 
## expects the path to a pickle file.)


## Most code written in this file will be DIY. It's important that you get to practice
## cleaning datasets and visualising them, so we purposely won't give you too much starter
## code. It'll be up to you to look up documentation and understand different Python modules.
## That being said, the task shouldn't be too hard, so we won't send you down any rabbit hole.

# function to clean the data automatically
def auto_op(my_file, size):
    dataObject = []
    for im in my_file:
        width, height = im.size
        if width != height:
            padding = abs(width-height)/2
            if width>height:
                im = crop(im, padding, 0, width-padding, height) #make new file
            else:
                im = crop(im, 0, padding, width, height-padding)
        if width != size:
            im = resize(im, size, size)
        dataObject.append(im)
    return dataObject

# function to clean the data by hand
def byhand_op(data_list, label_list):
    pre_data = []
    pre_label = []
    pre_data.append(data_list[0].rotate(270))
    pre_label.append(label_list[0])
    pre_data.append(data_list[1].rotate(90))
    pre_label.append(0)
    pre_data.append(data_list[2].rotate(180))
    pre_label.append(label_list[2])
    pre_data.append(data_list[3].rotate(180))
    pre_label.append(3)
    pre_data.append(data_list[4])
    pre_label.append(3)
    pre_data.append(data_list[5].rotate(180))
    pre_label.append(2)
    pre_data.append(data_list[6].rotate(315))
    pre_label.append(7)
    pre_data.append(data_list[7].rotate(180))
    pre_label.append(label_list[7])
    pre_data.append(data_list[8].rotate(315))
    pre_label.append(5)
    pre_data.append(data_list[9].rotate(315))
    pre_label.append(5)
    pre_data.append(data_list[10].rotate(315))
    pre_label.append(label_list[10])
    pre_data.append(data_list[11].rotate(90))
    pre_label.append(9)
    pre_data.append(data_list[12].rotate(180))
    pre_label.append(label_list[12])
    pre_data.append(data_list[13].rotate(90))
    pre_label.append(7)
    pre_data.append(data_list[14].rotate(180))
    pre_label.append(7)
    pre_data.append(data_list[15].rotate(315))
    pre_label.append(9)
    pre_data.append(data_list[16].rotate(315))
    pre_label.append(label_list[16])
    pre_data.append(data_list[17].rotate(315))
    pre_label.append(0)
    pre_data.append(data_list[18].rotate(90))
    pre_label.append(2)
    pre_data.append(data_list[19].rotate(90))
    pre_label.append(label_list[19])
    pre_data.append(data_list[20].rotate(315))
    pre_label.append(3)
    pre_data.append(data_list[21].rotate(315))
    pre_label.append(1)
    pre_data.append(data_list[22].rotate(180))
    pre_label.append(4)
    pre_data.append(data_list[23].rotate(135))
    pre_label.append(8)
    pre_data.append(data_list[24].rotate(90))
    pre_label.append(2)
    pre_data.append(data_list[25].rotate(90))
    pre_label.append(3)
    pre_data.append(data_list[26].rotate(90))
    pre_label.append(0)
    pre_data.append(data_list[27].rotate(180))
    pre_label.append(2)
    pre_data.append(data_list[28].rotate(315))
    pre_label.append(4)
    pre_data.append(data_list[29].rotate(90))
    pre_label.append(2)
    pre_data.append(data_list[30].rotate(90))
    pre_label.append(label_list[30])
    pre_data.append(data_list[31].rotate(90))
    pre_label.append(label_list[31])
    pre_data.append(data_list[32].rotate(315))
    pre_label.append(label_list[32])
    pre_data.append(data_list[33].rotate(315))
    pre_label.append(label_list[33])
    pre_data.append(data_list[34].rotate(90))
    pre_label.append(label_list[34])
    pre_data.append(data_list[35].rotate(270))
    pre_label.append(8)
    pre_data.append(data_list[36].rotate(315))
    pre_label.append(9)
    pre_data.append(data_list[37].rotate(180))
    pre_label.append(2)
    pre_data.append(data_list[38].rotate(270))
    pre_label.append(1)
    pre_data.append(data_list[39].rotate(270))
    pre_label.append(6)
    pre_data.append(data_list[40].rotate(315))
    pre_label.append(label_list[40])
    pre_data.append(data_list[41].rotate(180))
    pre_label.append(7)
    pre_data.append(data_list[42].rotate(90))
    pre_label.append(9)
    pre_data.append(data_list[43].rotate(270))
    pre_label.append(5)
    pre_data.append(data_list[44].rotate(270))
    pre_label.append(9)
    pre_data.append(data_list[45].rotate(180))
    pre_label.append(label_list[45])
    pre_data.append(data_list[46].rotate(180))
    pre_label.append(7)
    pre_data.append(data_list[47].rotate(270))
    pre_label.append(label_list[47])
    pre_data.append(data_list[48].rotate(180))
    pre_label.append(0)
    pre_data.append(data_list[49].rotate(315))
    pre_label.append(label_list[49])
    pre_data.append(data_list[50].rotate(90))
    pre_label.append(label_list[50])
    pre_data.append(data_list[51].rotate(90))
    pre_label.append(3)
    pre_data.append(data_list[52].rotate(180))
    pre_label.append(label_list[52])
    pre_data.append(data_list[53].rotate(180))
    pre_label.append(2)
    pre_data.append(data_list[54].rotate(180))
    pre_label.append(label_list[54])
    pre_data.append(data_list[55].rotate(90))
    pre_label.append(0)
    pre_data.append(data_list[56].rotate(315))
    pre_label.append(label_list[56])
    pre_data.append(data_list[57].rotate(180))
    pre_label.append(8)
    pre_data.append(data_list[58].rotate(90))
    pre_label.append(label_list[58])
    pre_data.append(data_list[59].rotate(90))
    pre_label.append(label_list[59])
    return pre_data, pre_label

if __name__ == "__main__":
    ## Running this script should read the input images.pkl and labels.pkl and clean the data
    ## and store cleaned data into the data/train and data/val folders
    ## To correct rotated images and add missing labels, you might want to prompt the terminal
    ## for input, so that you can input the angle and the missing label
    ## Remember, the first 60 images are rotated, and might contain missing labels.

    #clean the data for first 60 images by hand
    data_list = load_pickle_file('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a3/data/data/images.pkl')
    label_list = load_pickle_file('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a3/data/data/labels.pkl')
    pre_data, pre_label = byhand_op(data_list, label_list)
    #auto process the images after 60
    new_data_list = data_list[60:]
    new_label_list = label_list[60:]
    new_data_list = auto_op(new_data_list, 28)
    # combine and dump two parts into corresponding file
    new_data = open('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a3/data/data/cleaned_data.pkl', 'wb')
    new_label = open('/Users/wendyyyy/Cornell/CDS/IntSys-Education-master/a3/data/data/cleaned_label.pkl', 'wb')
    pickle.dump(pre_data+new_data_list,new_data)
    pickle.dump(pre_label+new_label_list,new_label)
    new_data.close()
    new_label.close()

