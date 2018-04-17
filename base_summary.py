import pandas as pd

class Summary(object):

    def __init__(self, dataset):

        self.dataset = dataset
        self.columns = self.dataset.columns

        # 基本特征属性判断，在基类中实现
        self.categroyFeatures, self.digitalFeatures, self.timeFeature = self._columnsRecommend()

    def _columnsRecommend(self):

        categroyFeatures = []
        digitalFeatures = []
        timeFeature = None

        for feature in self.columns:
            try:
                self.dataset[feature] = pd.to_datetime(self.dataset[feature])  # 这里默认只有一个维度是time feature
                timeFeature = feature
                break
            except:
                continue

        for feature in self.columns:

            if feature == timeFeature:
                continue
            elif self.dataset[feature].dtype == object:
                self.dataset[feature].fillna("null", inplace=True)
                categroyFeatures.append(feature)
            elif "id" in feature.lower() or "type" in feature.lower() or "code" in feature.lower() or "label" in feature.lower(): # 这里只是简单rule是匹配，后续要精细化
                categroyFeatures.append(feature)
            else:
                digitalFeatures.append(feature)

        return categroyFeatures, digitalFeatures, timeFeature

    def getColumnRecommend(self):

        recommendDict = {}
        for feature in self.categroyFeatures:
            recommendDict[feature] = "category"
        for feature in self.digitalFeatures:
            recommendDict[feature] = "digital"
        recommendDict[self.timeFeature] = "time"

        return recommendDict

    def cleanData(self, savePath):

        self.dataset['errorDetails'] = self.dataset['errorDetails'].apply(lambda x: x.replace('\r\n', ''))
        self.dataset.to_csv(savePath, index=False)
