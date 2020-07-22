import flask
import uuid
import os
from flask import request, jsonify
from functions import getFileBase64, getOcr, readTextFileContents, getSuggestedTags, getDictKey, fileExists, generateSessionToken, getCosineSimilarity
from conn import Conn
from classifier import Classifier
from strings import *

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['CORS_HEADERS'] = 'Content-Type'

_labelAll = ["negative", "neutral", "positive"]


def getRequiredParameters(_request, **kwargs):
    """
    Checks if the request body contains all the required parameters. Uses a hack with **kwargs.

    :param _request: The request body.
    :param kwargs: The required parameters.
    :return: Returns False and the parameter that was missing if there was missing data. Returns True and a dict containing all the parameter name and their values if successful.
    """
    if hasattr(_request, "json"):
        requestReceived = _request.json
    else:
        requestReceived = _request

    for key, value in kwargs.items():
        kwargs[key] = requestReceived.get(key)

    if None in kwargs.values():  # if one parameter is missing
        return [False, getDictKey(kwargs, kwargs.get(None))]
    else:
        return [True, kwargs]


@app.route('/api/login', methods=['POST'])
def login():
    """
    Checks user's credentials and if successful, returns a session token lasting 3 hours for the user.

    Required params:\n
    -action = 'login'\n
    -username\n
    -userpw\n

    :return: A session token and 201 if successful.
    """
    request.get_data()
    if request.json is None:
        return jsonify(
            {
                'status':
                    {'error': True,
                     'message': "Incorrect header type. Header type should be application/json."},
                'data': None
            }), 400

    chkMissingData, data = getRequiredParameters(request, action='', username='', userpw='')

    if not chkMissingData:
        return jsonify(
            {
                'status':
                    {'message': 'Missing required parameter \'' + data + '\'',
                     'error': True},
                'data': None
            }), 400

    # assume password sent is always unhashed, unless specified
    data['hashed'] = False
    if request.json['hashed'] is not None and request.json['hashed']:
        data['hashed'] = request.json['hashed']

    con = Conn()
    sessionToken = generateSessionToken()

    if con.userLogin(data['username'], data['userpw'], sessionToken, data['hashed']) == -1:
        return jsonify(
            {
                'status':
                    {'error': False,
                     'message': "User login failed."},
                'data': None
            }), 403

    return jsonify(
        {
            'status': RETURN_SUCCESS_STATUS,
            'data': {'token': sessionToken}
        }), 201


@app.route('/api/registration', methods=['POST'])
def register():
    """
    Registers a user and creates a model for them.

    Required params:\n
    -action = 'register'\n
    -username\n
    -userpw\n
    -email\n
    -firstname\n
    -lastname\n
    :return: Returns the modelID and 201 for them.
    """
    request.get_data()
    if request.json is None:
        return jsonify(
            {
                'status':
                    {'error': True,
                     'message': "Incorrect header type. Header type should be application/json."},
                'data': None
            }), 400

    chkMissingData, data = getRequiredParameters(request, action='', username='', userpw='', email='',
                                                 firstname='', lastname='')

    if not chkMissingData:
        return jsonify(
            {
                'status':
                    {'message': 'Missing required parameter \'' + data + '\'',
                     'error': True},
                'data': None
            }), 400

    con = Conn()

    if data['action'] != 'register':
        return jsonify(
            {
                'status':
                    {'message': 'Wrong action. This endpoint is only for registration.',
                     'error': True},
                'data': None
            }), 403

    username = data["username"]
    password = data["userpw"]  # assume plain password sent over https

    if con.usernameExists(username):
        return jsonify(
            {
                'status':
                    {'message': 'Username already exists.',
                     'error': True},
                'data': None
            }), 403

    userID = con.registerUser(username=username, pwd=password, firstName=data['firstname'], lastName=data['lastname'],
                              email=data['email'], usePwHash=False)

    # make new model
    model = Classifier(makeNewModel=True)
    mid = con.addNewModel(userID=str(userID), modelName=model.modelName)

    # if error during user creation
    if userID == -1 or mid == -1:
        return jsonify(
            {
                'status':
                    {'message': 'Server encountered error with user registration. Please try again later',
                     'error': True},
                'data': None
            }), 500

    # if user registration is ok
    return jsonify(
        {
            'status': RETURN_SUCCESS_STATUS,
            'data': {'mid': mid}
        }), 201


@app.route('/api/retrain', methods=['PUT'])
def retrain():
    """
    Used to retrain the model.

    Required params:\n
    -action = 'retrain'\n
    -stoken => the session token \n
    -data => a list of trainingID that will be deleted\n

    :return: A success message and 200.
    """
    request.get_data()
    if request.json is None:
        return jsonify(
            {
                'status':
                    {'error': True,
                     'message': "Incorrect header type. Header type should be application/json."},
                'data': None
            }), 400

    chkMissingData, data = getRequiredParameters(request, stoken='', action='', data='')

    if not chkMissingData:
        return jsonify(
            {
                'status':
                    {'message': 'Missing required parameter \'' + data + '\'',
                     'error': True},
                'data': None
            }), 400

    con = Conn()
    userID = con.checkToken(data["stoken"])

    if userID == -1:
        return jsonify(INVALID_SESSION_TOKEN), 403

    if data['action'] != 'retrain':
        return jsonify(
            {
                'status':
                    {'message': 'Wrong action. This endpoint is only for retraining.',
                     'error': True},
                'data': None
            }), 403

    if type(data['data']) != list:
        return jsonify(
            {
                'status':
                    {'message': 'Bad data. Data in the \'data\' field must be sent as list.',
                     'error': True},
                'data': None
            }), 400

    # get old model info
    oldModelID, oldModelName = con.getModelInfo(str(userID))

    if not con.removeTraining(oldModelID, data['data']):
        return jsonify(
            {
                'status':
                    {'message': 'Internal server encountered. Please try again.',
                     'error': True},
                'data': None
            }), 500

    availableTraining = con.getAvailableTraining(oldModelID)

    # make a new model
    model = Classifier(makeNewModel=True)
    for eachData in availableTraining:
        model.train(text=eachData[2], sentiment=eachData[3], tags=eachData[4], fromDB=True)

    # update db with new model info
    if not con.updateModelInfo(oldModelID, modelName=model.modelName):
        return jsonify(
            {
                'status':
                    {'message': 'Internal server error when updating the resource.',
                     'error': True},
                'data': None
            }), 500

    if not oldModelName.endswith(PICKLE_FILE_EXTENSION):
        oldModelName += PICKLE_FILE_EXTENSION

    os.remove(os.path.join(os.path.dirname(__file__), 'Models/' + oldModelName))

    return jsonify(
        {
            'status':
                {'message': 'Model successfully retrained.',
                 'error': False},
            'data': None
        }), 200


@app.route('/api/training', methods=['POST'])
def getTrainingData():
    """
    Gets all the available training data.

    Required params:\n
    -action = 'getData'\n
    -stoken\n

    :return: Returns the training data for that user and 200.
    """
    request.get_data()
    if request.json is None:
        return jsonify(
            {
                'status':
                    {'error': True,
                     'message': "Incorrect header type. Header type should be application/json."},
                'data': None
            }), 400

    chkMissingData, data = getRequiredParameters(request, stoken='', action='')

    if not chkMissingData:
        return jsonify(
            {
                'status':
                    {'message': 'Missing required parameter \'' + data + '\'',
                     'error': True},
                'data': None
            }), 400

    con = Conn()
    userID = con.checkToken(data["stoken"])

    if userID == -1:
        return jsonify(INVALID_SESSION_TOKEN), 403

    if data['action'] != 'getData':
        return jsonify(
            {
                'status':
                    {'message': 'Wrong action. This endpoint is only for retrieving training data.',
                     'error': True},
                'data': None
            }), 403

    modelID, modelName = con.getModelInfo(str(userID))

    availableTraining = con.getAvailableTraining(modelID)

    formattedTraining = list()
    for item in availableTraining:
        trainingDict = dict()
        trainingDict["id"] = item[0]
        trainingDict["rawText"] = item[1]
        trainingDict["sentiment"] = item[3]
        trainingDict["tags"] = item[4]
        trainingDict["dateTrained"] = item[5]
        formattedTraining.append(trainingDict)

    return jsonify(
        {
            'status': RETURN_SUCCESS_STATUS,
            'data':
                {
                    'availableTraining': formattedTraining}
        }), 200


@app.route('/api/ocr', methods=['POST'])
def getOCRtext():
    """
    Gets extracted text for an image.

    Required params:\n
    -action = 'ocr'\n
    -stoken\n
    -datatype = blob\n
    -data => list (if multiple) or string (if single)\n

    :return: Returns the extracted text and 200.
    """
    request.get_data()
    if request.json is None:
        return jsonify(
            {
                'status':
                    {'error': True,
                     'message': "Incorrect header type. Header type should be application/json."},
                'data': None
            }), 400

    data = request.json
    noMissingData, data = getRequiredParameters(request, stoken='', datatype='', data='', action='')

    if not noMissingData:
        return jsonify(
            {
                'status':
                    {'message': 'Missing required parameter \'' + data + '\'',
                     'error': True},
                'data': None
            }), 400

    con = Conn()
    userID = con.checkToken(data['stoken'])

    if userID == -1:
        return jsonify(INVALID_SESSION_TOKEN), 403

    if data['action'] != 'ocr':
        return jsonify(
            {
                'status':
                    {'message': 'Wrong action. This endpoint is only for getting OCR text.',
                     'error': True},
                'data': None
            }), 403

    if data['datatype'] != 'blob':
        return jsonify(
            {
                'status':
                    {'message': 'Wrong request. This endpoint is only for blobs.',
                     'error': True},
                'data': None
            }), 403

    if type(data['data']) != list and type(data['data']) != str:
        return jsonify(
            {
                'status':
                    {'message': 'Wrong data type for \'data\' field. Accepted types are str or list.',
                     'error': True},
                'data': None
            }), 400

    if type(data['data']) == list:  # if multiple files
        status, files = [], []
        for item in data['data']:
            filename = str(uuid.uuid4())  # generate temp file name
            iferr, retval = getFileBase64(item, filename, userID)
            status.append(iferr)
            files.append(retval)

        if False in status:  # if any error during converting the file
            return jsonify(
                {
                    'status':
                        {'error': False,
                         'message': ERRORFILEEXTEN}
                }), 400

        ocrtext = []
        for eachFile in files:
            ocrtext.append(getOcr(eachFile))
            os.remove(eachFile)

        return jsonify(
            {
                'status': RETURN_SUCCESS_STATUS,
                'data':
                    {
                        'ocrtext': ocrtext,
                        'type': 'list'}
            }), 200

    elif type(data['data']) == str:  # if single file
        filename = str(uuid.uuid4())  # generate temp file name
        status, filepath = getFileBase64(request.json['data'], filename, userID)

        if not status:
            # if error during converting the file
            return jsonify(
                {
                    'status':
                        {'error': False,
                         'message': ERRORFILEEXTEN}
                }), 400

        ocrtext = getOcr(filepath)
        os.remove(filepath)  # remove temp file
        return jsonify(
            {
                'status': RETURN_SUCCESS_STATUS,
                'data':
                    {
                        'ocrtext': ocrtext,
                        'type': 'str'}
            }), 200


@app.route('/api/suggestedTags', methods=['GET'])
def getTags():
    """
    Gets suggested tags.

    Required params:\n
    -tags => list of tags, separated by a delimiter (default commas)
    -mid => model ID

    Optional params:\n
    -howMany => default 5
    -delim => default comma, used to separate the tags
    :return: Suggested tags based on previous taught knowledge.
    """
    request.get_data()
    if request.args is None:
        return jsonify(
            {
                'status':
                    {'error': True,
                     'message': "No data is received."},
                'data': None
            }), 400

    userTags = request.args.get("tags")
    modelID = request.args.get("mid")
    returnNumber = request.args.get("howMany") if request.args.get("howMany") is not None and int(
        request.args.get("howMany")) > 0 else DEFAULT_RETURN_SUGGESTED_TAGS
    delim = request.args.get("delim") if request.args.get("delim") is not None else ","

    if userTags is None or modelID is None:
        return jsonify(
            {
                'status':
                    {'error': True,
                     'message': "Incomplete arguments. Expected mid and tags but only found " + (
                         "usertags" if modelID is None else "modelID")},
                'data': None
            }), 400

    if type(userTags) == str:
        userTags = list(userTags.split(delim))
    elif type(userTags) != list:
        return jsonify(
            {
                'status':
                    {'error': True,
                     'message': "Wrong format for tags. Allowed formats are either string or lists in square brackets."},
                'data': None
            }), 400

    con = Conn()
    tagsFromDB = con.getTags(modelID=modelID)

    relatedTags = getSuggestedTags(userTags, tagsFromDB, returnNumber)

    return jsonify(
        {
            'status': RETURN_SUCCESS_STATUS,
            'data': {"tags": userTags,
                     "mid": modelID,
                     "suggested": relatedTags,
                     "howMany": returnNumber}
        }), 200


@app.route('/api/train', methods=['POST'])
def processTeach():
    """
    Teaches the model with new knowledge.

    Required params:\n
    -action = 'teach'\n
    -stoken\n
    -datatype => text or blob\n
    -data => raw text or base64 blob\n
    -sentiment => 'positive', 'negative' or 'neutral'\n
    -tags => list of tags\n
    :return: A success message and 201
    """
    request.get_data()
    if request.json is None:
        return jsonify(
            {
                'status':
                    {'error': True,
                     'message': "Incorrect header type. Header type should be application/json."},
                'data': None
            }), 400

    data = request.json
    """noMissingData, data = getRequiredParameters(request, stoken='', action='', datatype='', data='', sentiment='', tags='')

    if not noMissingData:
        return jsonify(
            {
                'status':
                    {'message': 'Missing required parameter \'' + data + '\'',
                     'error': True},
                'data': None
            }), 400"""

    con = Conn()
    userID = con.checkToken(data['stoken'])

    if userID == -1:
        return jsonify(INVALID_SESSION_TOKEN), 403

    if data['action'] != "teach":
        return jsonify(
            {
                'status':
                    {'message': 'Wrong action. This endpoint is only for teach.',
                     'error': True},
                'data': None
            }), 403

    if data['datatype'] != 'text' and data['datatype'] != 'blob':
        return jsonify(
            {
                'status':
                    {
                        'message': 'Wrong datatype. This endpoint only accepts raw text as \'text\' and pdf or txt files as \'blob\'.',
                        'error': True},
                'data': None
            }), 403

    if data['datatype'] == 'text':
        txtdata = str(data['data'])
        tags = data['tags']
        sentiment = data['sentiment']

        # remove escape characters, if any
        if type(txtdata) is str:
            txtdata = txtdata.replace("\\", "")

        if sentiment not in _labelAll:
            return jsonify(
                {
                    'status':
                        {
                            'message': "Error in sentiments.",
                            'error': True
                        },
                    'data': None
                }), 400

        if con.getModelInfo(str(userID)) == [None, None]:
            return jsonify(
                {
                    'status':
                        {
                            'message': "No model created yet.",
                            'error': True
                        },
                    'data': None
                }), 400

        modelID, modelName = con.getModelInfo(str(userID))
        model = Classifier(makeNewModel=False, modelName=modelName)
        processedText = model.train(text=txtdata, sentiment=data['sentiment'], tags=tags, returnProcessedText=True)

        con.addNewTraining(modelID=modelID, tags=tags, sentiment=data['sentiment'], processedText=processedText,
                           rawText=txtdata)

        return jsonify(
            {
                'status': RETURN_SUCCESS_STATUS,
                'data': None
            }), 201
    elif data['datatype'] == 'blob':
        if type(data['data']) != list and type(data['data']) != str:
            return jsonify(
                {
                    'status':
                        {'error': True,
                         'message': 'Data in the \'data\' field should be sent as a string or a list of items.'},
                    'data': None
                }), 400

        if type(data['data']) == list:  # if multiple files
            status, files = list(), list()
            for item in data['data']:
                filename = str(uuid.uuid4())  # generate temp file name
                # retval is the filename when iferr = False or the error message when iferr = True
                iferr, retval = getFileBase64(item, filename, userID)

                # in case file received is not a text-based (.pdf/.txt) file
                if not retval.endswith('pdf') and not retval.endswith('txt'):
                    return jsonify(
                        {
                            'status':
                                {'error': True,
                                 'message': "Wrong format for data."},
                            'data': None
                        }), 400

                status.append(iferr)
                files.append(retval)

            if False in status:  # if any error during converting the file
                return jsonify(
                    {
                        'status':
                            {'error': False,
                             'message': ERRORFILEEXTEN}
                    }), 400

            modelID, modelName = con.getModelInfo(str(userID))
            model = Classifier(makeNewModel=False, modelName=modelName)

            tags = data['tags']

            # get text for each file
            for eachFile in files:
                txtdata = readTextFileContents(eachFile, return_metadata=False)
                os.remove(eachFile)  # remove temp file
                processedText = model.train(text=txtdata, sentiment=data['sentiment'], tags=tags, returnProcessedText=True)
                con.addNewTraining(modelID=modelID, tags=tags, sentiment=data['sentiment'],
                                   processedText=processedText, rawText=txtdata)

            return jsonify(
                {
                    'status': RETURN_SUCCESS_STATUS,
                    'data': None
                }), 201
        elif type(data['data']) == str:  # if single file
            filename = str(uuid.uuid4())  # generate temp file name
            status, filepath = getFileBase64(request.json['data'], filename, userID)

            # in case file received is not a text-based (.pdf/.txt) file
            if not filepath.endswith('pdf') and not filepath.endswith('txt'):
                return jsonify(
                    {
                        'status':
                            {'error': True,
                             'message': "Wrong format for data."},
                        'data': None
                    }), 400

            if not status:
                return jsonify(
                    {
                        'status':
                            {'error': True,
                             'message': ERRORFILEEXTEN},
                        'data': None
                    }), 400

            modelID, modelName = con.getModelInfo(str(userID))
            model = Classifier(makeNewModel=False, modelName=modelName)

            tags = data['tags']

            txtdata = readTextFileContents(filepath, return_metadata=False)
            os.remove(filepath)  # remove temp file
            processedText = model.train(text=txtdata, sentiment=data['sentiment'], tags=tags,
                                        returnProcessedText=True)
            con.addNewTraining(modelID=modelID, tags=tags, sentiment=data['sentiment'],
                               processedText=processedText, rawText=txtdata)
            return jsonify(
                {
                    'status': RETURN_SUCCESS_STATUS,
                    'data': None
                }), 200


@app.route('/api/predict', methods=['POST'])
def processAsk():
    """
    Gets a prediction from the model.

    Required params:\n
    -action = 'ask'\n
    -stoken\n
    -datatype => text or blob\n
    -data => raw text or base64 blob\n
    -tags => list of tags\n
    :return: The predicted sentiment, related tags and the original submitted text and 201
    """
    request.get_data()
    if request.json is None:
        return jsonify(
            {
                'status':
                    {'error': True,
                     'message': "Incorrect header type. Header type should be application/json."},
                'data': None
            }), 400

    data = request.json
    noMissingData, data = getRequiredParameters(request, stoken='', action='', datatype='', data='')

    if not noMissingData:
        return jsonify(
            {
                'status':
                    {'message': 'Missing required parameter \'' + data + '\'',
                     'error': True},
                'data': None
            }), 400

    con = Conn()
    userID = con.checkToken(data['stoken'])

    if userID == -1:
        return jsonify(INVALID_SESSION_TOKEN), 403

    if data['action'] != "ask":
        return jsonify(
            {
                'status':
                    {'message': 'Wrong action. This endpoint is only for teach.',
                     'error': True},
                'data': None
            }), 403

    if data['datatype'] != 'text' and data['datatype'] != 'blob':
        return jsonify(
            {
                'status':
                    {
                        'message': 'Wrong datatype. This endpoint only accepts raw text as \'text\' and pdf or txt files as \'blob\'.',
                        'error': True},
                'data': None
            }), 403

    if data['datatype'] == 'text':
        txtdata = str(request.json['data'])

        # remove escape characters, if any
        txtdata = txtdata.replace("\\", "")

        if con.getModelInfo(str(userID)) == [None, None]:
            return jsonify(
                {
                    'status':
                        {
                            'message': "No model created yet.",
                            'error': True
                        },
                    'data': None
                }), 400

        modelID, modelName = con.getModelInfo(str(userID))

        availTraining = con.getAvailableTraining(modelID)

        ifPrevKnowSimilar = False
        for eachTraining in availTraining:
            if getCosineSimilarity(txtdata, eachTraining[1]) >= PREVIOUS_KNOWLEDGE_SIMILARITY_RATE:
                ifPrevKnowSimilar = True
                break

        if not ifPrevKnowSimilar:
            return jsonify(
                {
                    'status': RETURN_SUCCESS_STATUS,
                    'data':
                        {
                            'predictedSentiment': "Not available",
                            'suggested': "Not available",
                            'text': txtdata
                        }
                }), 200

        model = Classifier(makeNewModel=False, modelName=modelName)

        predSentiment, retTags = model.predict(txtdata)

        return jsonify(
            {
                'status': RETURN_SUCCESS_STATUS,
                'data':
                    {
                        'predictedSentiment': predSentiment,
                        'suggested': retTags,
                        'text': txtdata
                    }
            }), 200
    elif data['datatype'] == 'blob':
        if type(data['data']) != list and type(data['data']) != str:
            return jsonify(
                {
                    'status':
                        {'error': True,
                         'message': 'Data in the \'data\' field should be sent as a string or a list of items.'},
                    'data': None
                }), 400

        if type(data['data']) == list:  # if multiple files
            modelID, modelName = con.getModelInfo(str(userID))
            model = Classifier(makeNewModel=False, modelName=modelName)

            predSentiment = list()
            retTags = list()
            for item in data['data']:
                filename = str(uuid.uuid4())  # generate temp file name
                status, filepath = getFileBase64(item, filename)

                # in case file received is not a text-based (.pdf/.txt) file
                if not filepath.endswith('pdf') and not filepath.endswith('txt'):
                    os.remove(filepath)
                    return jsonify(
                        {
                            'status':
                                {'error': True,
                                 'message': "Wrong format for data."},
                            'data': None
                        }), 400

                if not status:
                    return jsonify(
                        {
                            'status':
                                {'error': False,
                                 'message': ERRORFILEEXTEN}
                        }), 400

                textData = readTextFileContents(filepath)
                sent, tag = model.predict(textData)
                predSentiment.append(sent)
                retTags.append(tag)
                os.remove(filepath)

            return jsonify(
                {
                    'status': RETURN_SUCCESS_STATUS,
                    'data':
                        {'predictedSentiment': predSentiment,
                         'suggested': retTags,
                         'text': data['data']}
                }), 200
        elif type(data['data']) == str:  # if single file
            modelID, modelName = con.getModelInfo(str(userID))
            model = Classifier(makeNewModel=False, modelName=modelName)

            filename = str(uuid.uuid4())  # generate temp file name
            status, filepath = getFileBase64(data['data'], filename)

            # in case file received is not a text-based (.pdf/.txt) file
            if not filepath.endswith('pdf') and not filepath.endswith('txt'):
                os.remove(filepath)
                return jsonify(
                    {
                        'status':
                            {'error': True,
                             'message': "Wrong format for data."},
                        'data': None
                    }), 400

            textData = readTextFileContents(filepath)
            os.remove(filepath)

            predSentiment, retTags = model.predict(textData)

            return jsonify(
                {
                    'status': RETURN_SUCCESS_STATUS,
                    'data':
                        {'predictedSentiment': predSentiment,
                         'suggested': retTags,
                         'text': data['text']}
                }), 200


if __name__ == '__main__':
    if fileExists("./Files/tika.pdf"):
        readTextFileContents("./Files/tika.pdf")

    app.run(host="0.0.0.0")
