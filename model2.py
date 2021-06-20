import csv
import joblib
print(chr(27) + "[2J")

loaded_model = joblib.load('random_forest.sav')
print("Loaded Successfully")
# Creating dictionaries for each encoded variable

reader = csv.reader(open('encoded_files\district_encoded.csv', 'r'))
dict_district = {}
for row in reader:
    k = row[1]
    v = row[2]
    dict_district[k] = v

del dict_district['district']
# print(dict_district)
reader2 = csv.reader(open('encoded_files\crop_encoded.csv', 'r'))
dict_crop = {}
for row in reader2:
    a = row[1]
    b = row[2]
    dict_crop[a] = b

del dict_crop['crop']
# print(dict_crop)
reader3 = csv.reader(open('encoded_files\season_encoded.csv', 'r'))
dict_season = {}
for row in reader3:
    x = row[1]
    y = row[2]
    dict_season[x] = y

del dict_season['season']
# print(dict_season)


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"


# Prediction
# predDistrict, predYear, predCrop, predSeason, predArea, predRain, predTemp = 0, 0, 0, 0, 0, 0, 0
# District
print(dict_district)
predDistrict = input("Enter District number:")
print("You entered ", get_key(predDistrict, dict_district))
predDistrict = int(predDistrict)

# Year
predYear = int(input("Enter Year:"))
print("You entered ", predYear)

# Crop
# print(dict_crop)
# predCrop = input("Enter crop number:")
# print("You entered ", get_key(predCrop, dict_crop))
# predCrop = int(predCrop)

# Season
print(dict_season)
predSeason = input("Enter season number:")
print("You entered ", get_key(predSeason, dict_season))
predSeason = int(predSeason)

# Area
predArea = float(input("Enter Area(cm^2):"))
print("You entered ", predArea)

# Rain
predRain = float(input("Enter Rain(in mm):"))
print("You entered ", predRain)

# Temp
predTemp = float(input("Enter Temp(in degree celsius):"))
print("You entered ", predTemp)

# predList = []
# predList.append(predDistrict)
# predList.append(predYear)
# predList.append(predCrop)
# predList.append(predSeason)
# predList.append(predArea)
# predList.append(predRain)
# predList.append(predTemp)
# print(predList)

# predc = 10 ** loaded_model.predict(Xnew2)
# print(predc)

finalDict = {}
for x in range(21):
    Xnew2 = [[predDistrict, predYear, x,
              predSeason, predArea, predRain, predTemp]]
    predc = 10 ** loaded_model.predict(Xnew2)
    predcVal = int(predc[0])
    crop = get_key(str(x), dict_crop)
    # crop = int(crop)
    finalDict[crop] = predcVal
    # print(predc)
    Xnew2 *= 0

# {k: v for k, v in sorted(finalDict.items(), key=lambda item: item[1])}
marklist = sorted(finalDict.items(), key=lambda x: x[1])
sortdict = dict(marklist)
print(sortdict)
# print(finalDict)
keys_list = list(sortdict)

print("Crops with most Yield suitable with input parameters")
# Xnew2[0].pop(2)
# print(predDistrict, predYear, predSeason, predArea, predRain, predTemp)
print(keys_list[20])
print(keys_list[19])
print(keys_list[18])
