from Crypto.Hash import SHA512
import MySQLdb
from MySQLdb import MySQLError
from datetime import datetime


class Conn:
    """
    Wrapper class for performing all database-related queries.
    """
    _db: MySQLdb.connection = None
    isConnected: bool = False
    errorMessage: str = ""

    def __init__(self):
        try:
            self._db = MySQLdb.connect(host="localhost", user="root", passwd="", db="ezml")
            self.isConnected = True
        except MySQLError as e:
            self.errorMessage = repr(e)

    def __del__(self):
        if self.isConnected:
            self._db.close()

    def registerUser(self, username: str, pwd: str, firstName: str, lastName: str, email: str, usePwHash: bool = True) -> int:
        """
        Registers a user and returns the registered userID.

        :param username: The username to register on. Duplicate usernames are not checked here.
        :param pwd: The password of the user.
        :param firstName: The firstname of the user.
        :param lastName: The lastname of the user.
        :param email: The email of the user.
        :param usePwHash: False if the user uses their own hash. Included for compatibility with the web app. Recommended to default to True.
        :return: The userID of the registered user if operation is successful. Returns -1 otherwise.
        """
        if not self.isConnected:
            return -1

        if usePwHash:
            pwdHash = SHA512.new(pwd.encode()).hexdigest()
        else:
            pwdHash = pwd

        todayDate = str(datetime.now())
        pwdHash = SHA512.new(pwd.encode()).hexdigest()

        cur = self._db.cursor()

        try:
            cur.execute(
                "insert into users (firstName, lastName, username, password, email, registeredDate, lastLoginDate) values (%s, %s, %s, %s, %s, %s, %s);",
                (firstName, lastName, username, pwdHash, email, todayDate, todayDate))
            self._db.commit()
            userID = cur.lastrowid
        except MySQLError as e:
            self._db.rollback()
            userID = -1

        cur.close()
        return userID

    def userLogin(self, username: str, pwd: str, sessionToken: str, usePwHash: bool = True) -> int:
        """
        Performs user login using the session token generated. Then updates the lastLoginDate and the sessionToken field in the database.

        :param username: The username to check.
        :param pwd: The password to check.
        :param sessionToken: A 26 character Base64 encoded string. Use generateSessionToken() in functions to generate a session token.
        :param usePwHash: False if the user uses their own hash. Included for compatibility with the web app. Recommended to default to True.
        :return: The userID if login is successful. Returns -1 otherwise.
        """
        if not self.isConnected:
            return -1

        if usePwHash:
            pwdHash = SHA512.new(pwd.encode()).hexdigest()
        else:
            pwdHash = pwd
        cur = self._db.cursor()

        userID = -1
        try:
            cur.execute("select userid from users where username = %s and password = %s;", (username, pwdHash))

            result = cur.fetchall()

            if len(result) == 1:  # if username/pw is correct
                userID = int(result[0][0])
                try:
                    cur.execute("update users set lastLoginDate = %s, sessionToken = %s where userid = %s;", (datetime.now(), sessionToken, userID))
                    self._db.commit()
                except MySQLError:
                    self._db.rollback()
                    userID = -1
        except MySQLError:
            pass

        cur.close()
        return userID

    def checkToken(self, sessionToken: str) -> int:
        """
        Checks to see if the session token is still valid. Default is 3 hours.

        :param sessionToken: A 26 character Base64 encoded string. Use generateSessionToken() in functions to generate a session token.
        :return: The userID if token is still valid. -1 otherwise.
        """
        if not self.isConnected:
            return -1

        cur = self._db.cursor()

        userID: int = -1

        try:
            cur.execute("select userID, if(timediff(now(), (select lastLoginDate from users where sessionToken = %s)) < maketime(3,0,0), true, false) as 'isTokenValid' from users where sessionToken = %s", (sessionToken, sessionToken))

            result = cur.fetchall()

            if len(result) == 1:
                userID = int(result[0][0])
        except MySQLError:
            pass

        cur.close()
        return userID

    def _updatePassword(self, username: str, newPw: str) -> bool:
        """Unused function"""
        if not self.isConnected:
            return False

        cur = self._db.cursor()
        pwdHash = SHA512.new(newPw.encode()).hexdigest()

        retVal: bool = False
        try:
            cur.execute("update users set password = %s where username = %s", (pwdHash, username))
            self._db.commit()
            retVal = True
        except MySQLError:
            self._db.rollback()

        cur.close()
        return retVal

    def usernameExists(self, username: str) -> bool:
        """
        Checks if the username exists for a particular user.

        :param username: The username to check for.
        :return: True if the username already exists, false otherwise.
        """
        if not self.isConnected:
            return False

        cur = self._db.cursor()

        result = None
        retVal: bool = False
        try:
            cur.execute("select userid from users where username = %s;", (username,))
            result = cur.fetchall()

            if len(result) == 1:
                retVal = True
        except MySQLError:
            pass

        cur.close()

        return retVal

    def addNewModel(self, userID: str, modelName: str) -> int:
        """
        Adds a new model entry to the database.

        :param userID: The userID of the owner of the model.
        :param modelName: The name of the model.
        :return: The modelID if operation is successful, -1 otherwise.
        """
        if not self.isConnected:
            return -1

        cur = self._db.cursor()

        modelID: int = -1
        try:
            cur.execute("insert into user_model (userID, modelName, defaultModel) values (%s, %s, 1);",
                        (userID, modelName))
            self._db.commit()
            modelID = cur.lastrowid
        except MySQLError as e:
            self._db.rollback()

        cur.close()
        return modelID

    def getModelInfo(self, userID: str) -> tuple:
        if not self.isConnected:
            return (None, None)

        cur = self._db.cursor()

        modelID = None
        modelName = None
        try:
            cur.execute("select userOwnModelID, modelName from user_model where userID = %s and defaultModel = 1;", (userID,))
            result = cur.fetchall()
            if len(result) == 1:
                modelID = result[0][0]
                modelName = result[0][1]
        except MySQLError:
            pass

        cur.close()

        return modelID, modelName

    def updateModelInfo(self, modelID: str, modelName: str) -> bool:
        """
        Updates a model info (model name, specifically) in the database.

        :param modelID: The modelID of the model to be updated.
        :param modelName: The name of the model to be updated.
        :return:
        """
        if not self.isConnected:
            return False

        cur = self._db.cursor()

        retval = False
        try:
            cur.execute("update user_model set modelName = %s where userOwnModelID = %s;", (modelName, modelID))
            self._db.commit()
            retval = True
        except MySQLError:
            self._db.rollback()

        cur.close()
        return retval

    def getTags(self, modelID: str) -> tuple:
        """
        Gets all the tags based on all the available, non-deleted training data for a certain model.

        :param modelID: The modelID to get the training data.
        :return: A tuple of all the tags as strings. Returns None otherwise.
        """
        if not self.isConnected:
            return None

        cur = self._db.cursor()

        result = None
        try:
            cur.execute("select tags from training_data where userOwnModelID = %s and deleted = 0;", (modelID,))
            tmp = cur.fetchall()
            result = ()
            for item in tmp:
                result = result + item
        except MySQLError:
            pass

        cur.close()

        return result

    def getAvailableTraining(self, modelID: str) -> tuple:
        """
        Gets all the available (non-deleted) training for a model.

        :param modelID: The modelID to get the training data.
        :return: A tuple of all the results. Returns None otherwise.
        """
        if not self.isConnected:
            return None

        cur = self._db.cursor()

        result = None
        try:
            cur.execute("select trainingID, rawTextData, processedTextData, sentiment, tags, dateTrained from training_data where userOwnModelID = %s and deleted = 0 limit 10;", (modelID,))
            result = cur.fetchall()
        except MySQLError:
            pass

        cur.close()
        return result

    def addNewTraining(self, modelID: str, tags: list, sentiment: str, processedText: list, rawText: str) -> int:
        """
        Adds a new training to the training_data table.

        :param modelID: The ID of the model that was trained.
        :param tags: The tags as a list of words.
        :param sentiment: The sentiment, either 'positive', 'negative' or 'neutral'.
        :param processedText: The processed text. Set returnProcessedText = True to get this.
        :param rawText: The unprocessed text. Used for display purposes when user deletes the data.
        :return: The newly added trainingID if operation is successful. Returns -1 otherwise.
        """
        if not self.isConnected:
            return -1

        cur = self._db.cursor()
        trainingID: int = -1
        try:
            cur.execute(
            "insert into training_data (userOwnModelID, rawTextData, processedTextData, dateTrained, tags, sentiment, deleted) values (%s, %s, %s, %s, %s, %s, %s)",
            (modelID, rawText, str(processedText), str(datetime.now()), str(tags), sentiment, 0))

            self._db.commit()
            trainingID = cur.lastrowid
        except MySQLError:
            trainingID = -1
            self._db.rollback()

        cur.close()
        return trainingID

    def removeTraining(self, modelID, trainingID) -> bool:
        """
        Deletes (hides) the data that is to be deleted.

        :param modelID: A modelID to check if the trainingID does belong to the user.
        :param trainingID: A string or int of a data, or list or tuple if multiple data.
        :return: True if the operation is successful. False otherwise.
        """
        if not self.isConnected:
            return False

        cur = self._db.cursor()

        retVal: bool = False
        try:
            if type(trainingID) is list or type(trainingID) is tuple:
                query = "update training_data set deleted = 1 where userOwnModelID = " + str(modelID) + f" and trainingID in ({','.join(['%s'] * len(trainingID))})"
                cur.execute(query, trainingID)
            elif type(trainingID) is str or type(trainingID) is int:
                cur.execute("update training_data set deleted = 1 where userOwnModelID = %s and trainingID = %s;", (modelID, trainingID))

            self._db.commit()
            retVal = True
        except MySQLError as e:
            self._db.rollback()

        cur.close()

        return retVal
