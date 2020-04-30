"""
This file contains the necessary functions to make data augmentation, which consist of making
two geometric transformations of the existing data.   
Here, we process the existing data set we have, available under the Pascal VOC format. 
The goal is to enlarge this last using the “imgaug” library!
"""
import imgaug as ia
import cv2
import os
import imageio
import xml.etree.ElementTree as ET
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
from typing import List


# The path for the Pascal VOC data set
path_to_data_set = (
    "./data/training_images/cat_images/export/cat_faces--PascalVOC-export/"
)
# The path for the list of XML files
list_xml_files = os.listdir(os.path.join(path_to_data_set, "Annotations"))
# The path for the list of images (training and validation images )
list_jpg_images = os.listdir(os.path.join(path_to_data_set, "JPEGImages"))


def indent(elem: ET, level: int = 0) -> ET:
    """
    This function walks the tree and adds whitespace to the tree, 
    so that saving it as usual results in a prettyprinted tree.
    :param elem: it represents a node of the tree. 
    :param level: an integer representing the number of whitespace 
    :return: ET object
    """
    i = "\n" + level * "  "
    j = "\n" + (level - 1) * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem


def generate_xml(
    file_path: str,
    new_width: int,
    new_height: int,
    new_list_xmin: List[float],
    new_list_ymin: List[float],
    new_list_xmax: List[float],
    new_list_ymax: List[float],
    new_xml_file_name: str,
) -> None:
    """
    This function generates a new xml file from an existing one and it saves the new one in the same directory.
    :param file_path: the path to the existing xml file
    :param new_width: the new value of the tag width
    :param new_height: the new value of the tag height
    :param new_list_xmin: a list of new value(s) of the tag xmin
    :param new_list_ymin: a list of new value(s) of the tag ymin
    :param new_list_xmax: a list of new value(s) of the tag xmax
    :param new_list_ymax: a list of new value(s) of the tag ymax
    :param new_xml_file_name: the name of the new generated xml file
    :return: this function returns nothing  
    """
    # Sanity check
    # The four different lists must have the same length
    if (
        len(new_list_xmin) != len(new_list_ymin)
        or len(new_list_xmin) != len(new_list_ymax)
        or len(new_list_xmin) != len(new_list_xmax)
    ):
        raise Exception("The length of the four given list must be the same")

    # We start by parsing the existing xml file
    tree = ET.parse(file_path)
    root = tree.getroot()
    name_of_new_image = new_xml_file_name.split(".xml")[0]

    # We change the text of the node "filename"
    for x in root.iter("filename"):
        x.text = name_of_new_image + ".jpg"

    # We change the text of the node "path"
    for x in root.iter("path"):
        path = root[2].text
        path_without = path.split("/Annotations/")
        x.text = os.path.join(
            path_without[0], "Annotations", name_of_new_image + ".jpg"
        )

    # We change the text of the node "width"
    for x in root.iter("width"):
        x.text = str(new_width)

    # We change the text of the node "height"
    for x in root.iter("height"):
        x.text = str(new_height)

    # The counter "i" is for parsing the input lists
    i = 0
    # We change the content of the node "object"
    for x in root.iterfind("object"):
        x[4][0].text = str(new_list_xmin[i])
        x[4][1].text = str(new_list_ymin[i])
        x[4][2].text = str(new_list_xmax[i])
        x[4][3].text = str(new_list_ymax[i])
        i = +1

    # Now, we save the past modification in a new xml file
    tree = ET.ElementTree(indent(root))
    tree.write(os.path.join(path_to_data_set, "Annotations", new_xml_file_name))


def main():
    """
    This main function contains the necessary instructions to parse and create xml files,
    to make two geometric transformations to the existing images and to save the resulting ones using OpenCV library.
    At the end of the execution of this function, we get an enlarged Pascal VOC data set. 
    """

    # The path to the .txt file. This last contain the list of the training images.
    # So, the list of images we want to transform.
    path_to_txt_file = os.path.join(
        path_to_data_set, "ImageSets/Main/Cat_Face_train.txt"
    )
    # We open the .txt file in reading and writing mode.
    file_object = open(path_to_txt_file, "r+")

    # We parse this last line by line
    for line in file_object.readlines():
        # The name of the image in the given line
        image_name = line.split(".jpg")[0]
        # The path to its corresponding xml file
        xml_file_path = os.path.join(
            path_to_data_set, "Annotations", image_name + ".xml"
        )
        # We parse this last
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        # To read the corresponding image
        image = imageio.imread(
            os.path.join(path_to_data_set, "JPEGImages", image_name + ".jpg")
        )
        # We recover from the xml file the dimensions of the image
        width = int(root[4][0].text)
        height = int(root[4][1].text)
        # To resize the image
        image = ia.imresize_single_image(image, (height, width))

        # We create the empty lists that will contain the actual and the futur coordinates of the bounding box(es)
        list_xmin, new_list_xmin = [], []
        list_ymin, new_list_ymin = [], []
        list_xmax, new_list_xmax = [], []
        list_ymax, new_list_ymax = [], []
        list_bounding_boxes = []
        # We parse the xml file to get the coordinates of the bounding box(es)
        for i in root.iterfind("object"):
            if i[4][0].tag == "xmin":
                list_xmin.append(i[4][0].text)
            if i[4][1].tag == "ymin":
                list_ymin.append(i[4][1].text)
            if i[4][2].tag == "xmax":
                list_xmax.append(i[4][2].text)
            if i[4][3].tag == "ymax":
                list_ymax.append(i[4][3].text)
        # Sanity check
        if (
            len(list_xmin) != len(list_xmax)
            or len(list_ymin) != len(list_ymax)
            or len(list_xmin) != len(list_ymin)
        ):
            raise Exception("Something goes wrong in the lists of bounding boxes")

        # The number of bounding box(es) in the image
        number_of_bounding_boxes = len(list_ymin)
        # We create the list of bounding box(es) that we will use to creat an instance of the class BoundingBoxesOnImage
        for i in range(number_of_bounding_boxes):
            list_bounding_boxes.append(
                BoundingBox(
                    x1=float(list_xmin[i]),
                    x2=float(list_xmax[i]),
                    y1=float(list_ymin[i]),
                    y2=float(list_ymax[i]),
                )
            )
        # bbs is an instance of the class BoundingBoxesOnImage representing the boxes located on the image
        bbs = BoundingBoxesOnImage(list_bounding_boxes, shape=image.shape)

        # The first image augmentation that we want to apply
        seq = iaa.Sequential(
            [
                iaa.GammaContrast(1.5),
                iaa.Affine(translate_percent={"x": 0.1}, scale=0.8),
            ]
        )
        # bbs_aug is the object representing the bounding box(es) after the first augmentation
        # image_aug is the augmented image
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

        # The name we want to give to this augmented image
        name_of_new_image = image_name + "_1"
        filename_and_path_to_new_image = os.path.join(
            path_to_data_set, str("JPEGImages/" + name_of_new_image + ".jpg")
        )
        # To save this new image
        cv2.imwrite(filename_and_path_to_new_image, image_aug)

        # The list of new coordinates of the transformed bounding box(es)
        for i in range(len(bbs_aug.bounding_boxes)):
            new_list_xmin.append(bbs_aug.bounding_boxes[i].x1)
            new_list_ymin.append(bbs_aug.bounding_boxes[i].y1)
            new_list_xmax.append(bbs_aug.bounding_boxes[i].x2)
            new_list_ymax.append(bbs_aug.bounding_boxes[i].y2)

        # To creat the new xml file
        generate_xml(
            xml_file_path,
            image_aug.shape[1],
            image_aug.shape[0],
            new_list_xmin,
            new_list_ymin,
            new_list_xmax,
            new_list_ymax,
            name_of_new_image + ".xml",
        )

        # We add a new line in the file containing the name of the training images
        file_object.write("\n")
        file_object.write(name_of_new_image + ".jpg 1")

        # Now we are done with the first transformation
        # We define the second geometric transformation
        seq = iaa.Sequential(
            [
                iaa.CropAndPad(
                    percent=(-0.2, 0.2), pad_mode="edge"
                ),  # crop and pad images
                iaa.AddToHueAndSaturation((-60, 60)),  # change their color
                iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
            ],
            random_order=True,
        )

        # The obtained image and box(es) after the transformation
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        # The name we want to give to this new image
        name_of_new_image = image_name + "_2"
        filename_and_path_to_new_image = os.path.join(
            path_to_data_set, str("JPEGImages/" + name_of_new_image + ".jpg")
        )
        # Te save this new image
        cv2.imwrite(filename_and_path_to_new_image, image_aug)

        # the list of the new coordinates of these new transformed bounding box(es)
        new_list_xmin, new_list_ymin, new_list_xmax, new_list_ymax = [], [], [], []
        for i in range(len(bbs_aug.bounding_boxes)):
            new_list_xmin.append(bbs_aug.bounding_boxes[i].x1)
            new_list_ymin.append(bbs_aug.bounding_boxes[i].y1)
            new_list_xmax.append(bbs_aug.bounding_boxes[i].x2)
            new_list_ymax.append(bbs_aug.bounding_boxes[i].y2)

        # To create the corresponding xml file to this new image
        generate_xml(
            xml_file_path,
            image_aug.shape[1],
            image_aug.shape[0],
            new_list_xmin,
            new_list_ymin,
            new_list_xmax,
            new_list_ymax,
            name_of_new_image + ".xml",
        )
        # We add a new line with the name of image resulting from the last transformation
        file_object.write("\n")
        file_object.write(name_of_new_image + ".jpg 1")
    file_object.close()


if __name__ == "__main__":
    main()
