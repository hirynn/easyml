import os
from strings import TAGS_SIMILARITY_RATE, DEFAULT_RETURN_SUGGESTED_TAGS
import secrets
from tika import parser
from collections import Counter
import base64
import re
import strings
import math
import difflib
from PIL import Image
import pytesseract
import cv2

dirname = os.path.dirname(__file__)
tesseract_path = os.path.join(dirname, 'Tesseract/')
files_path = os.path.join(dirname, 'Files/')

if not os.path.exists(tesseract_path):
    os.mkdir(tesseract_path)
if not os.path.exists(files_path):
    os.mkdir(files_path)


# read all content from file, returns the content of the file, returns content
# also returns metadata (if available) if return_metadata is set to True
def readTextFileContents(_filename, return_metadata=False, encode="UTF-8"):
    if _filename.endswith(".txt") or _filename.endswith(".csv"):
        try:
            file = open(_filename, 'r', encoding=encode)
            text = file.read()
            return text
        except IOError as e:
            print("File open failed - Error: " + e)
    elif _filename.endswith(".pdf"):
        parsed = parser.from_file(_filename)
        # metadata = parsed['metadata']
        content = parsed['content']

        return content  # if not return_metadata else [content, metadata]


# gets the key from a key, value pair in a dict
def getDictKey(_dict, _val):
    for key, val in _dict.items():
        if _val == val:
            return key


def getVotes(_votes):
    count = Counter(_votes)
    return count.most_common(1)[0][0]


def getFileBase64(_stringData, _outFileName, _userID=None):
    if _userID is not None:  # make new folder for user based on their userID
        user_files = os.path.join(files_path, str(_userID) + "/")
        if not os.path.exists(user_files):
            os.mkdir(user_files)
    else:  # place outside otherwise
        user_files = files_path

    _pattern = "data\\:[\\w]+\\/[\\w]+\\;base64\\,"
    _discard, _base64Str = re.split(_pattern, _stringData, 1)
    _extension = re.search("data:[\\w]+/(\\w+)", _stringData)

    if _extension is not None:
        _b64DecodedStr = base64.b64decode(_base64Str)
        if _extension is "plain" or _extension.group(1) == "plain":  # .txt
            _filename = user_files + _outFileName + ".txt"
        else:
            _filename = user_files + _outFileName + "." + _extension.group(1)
        with open(_filename, "wb") as fh:
            fh.write(_b64DecodedStr)
        return [True, _filename]
    else:
        return [False, strings.ERRORFILEEXTEN]


def generateSessionToken():
    return secrets.token_urlsafe(26)


def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0) ** 2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0) ** 2 for k in terms))
    return dotprod / (magA * magB)


def _quicksort(arr: list, low: int, high: int):
    if low < high:
        pi = _partition(arr, low, high)
        _quicksort(arr, low, pi - 1)
        _quicksort(arr, pi + 1, high)


def _partition(arr: list, low: int, high: int):
    i = (low - 1)
    pivot = arr[high]

    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def _getListSimilarity(list1: list, list2: list) -> float:
    _quicksort(list1, 0, len(list1) - 1)
    _quicksort(list2, 0, len(list2) - 1)
    return difflib.SequenceMatcher(None, list1, list2).ratio()


def _getRelatedTags(userTagged, fromDB):
    if type(userTagged) == list:
        userTagged = set(userTagged)
    if type(fromDB) == list:
        fromDB = set(fromDB)

    return list(fromDB - userTagged)


def _separateTags(tagStr: str) -> list:
    """
    Gets concatenated string of tag IDs separated by delimiter character and returns list containing each item.

    :param tagStr: Concat tag as string, e.g. '['1', '3', '5', '4', '6', '8']'
    :return: List of tag as list, e.g. ['1', '3', '5', '4', '6', '8']
    """
    _pattern = "([\\w\\d\\ ]+)"
    return re.findall(_pattern, tagStr)


def toLower(list1: list):
    return [item.lower() for item in list1]


def getSuggestedTags(userTagged: list, tagsFromDB: list or tuple, returnNumOfTags=DEFAULT_RETURN_SUGGESTED_TAGS):
    if type(userTagged) == str:
        userTagged = [userTagged]

    if type(tagsFromDB) == tuple:
        tagsFromDB = list(tagsFromDB)

    relatedTags = []
    for item in tagsFromDB:
        item = _separateTags(item)
        item = list(filter(' '.__ne__, item))  # remove all blank spaces only
        if _getListSimilarity(toLower(userTagged), toLower(item)) >= TAGS_SIMILARITY_RATE:
            relatedTags.append(_getRelatedTags(userTagged, item))

    # get all unique tags (remove the ones tagged by user)
    finaltags = list()
    counter = 0
    for item in relatedTags:  # for all group of tags [[a,b],[c],...] retrieved from db
        if counter < int(returnNumOfTags):
            for tag in item:
                if tag not in finaltags:  # check for duplicates
                    finaltags.append(tag)
                    counter += 1
        else:
            break

    return finaltags


def getEpochIdentifier():
    import time
    import random
    epoch = time.time()
    return str(int(epoch)) + str(epoch - int(epoch))[random.randint(2, 8)]


def getOcr(imgPath: str, preProcessOptions: str = None) -> str:
    """
    Gets text from image using Tesseract
    :param imgPath: Path to image
    :param preProcessOptions: blur or thresh. Default is None.
    :return: OCR'd text
    """
    pytesseract.pytesseract.tesseract_cmd = tesseract_path + "tesseract.exe"
    # load the example image and convert it to grayscale
    image = cv2.imread(imgPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # check to see if we should apply thresholding to preprocess the
    # image
    if preProcessOptions == "thresh":
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # make a check to see if median blurring should be done to remove
    # noise
    elif preProcessOptions == "blur":
        gray = cv2.medianBlur(gray, 3)
    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)

    return text


def fileExists(_filename):
    try:
        file = open(_filename, 'r')
    except (IOError, FileNotFoundError, FileExistsError):
        return False

    return True
