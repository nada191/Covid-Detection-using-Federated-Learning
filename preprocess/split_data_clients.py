import os
import shutil


def distribute(input_path, output_path, client1, client2, client3, client4):
    covid = []
    normal = []
    for filename in os.listdir(input_path + "/covid"):
        covid.append(filename)
    for filename in os.listdir(input_path + "/normal"):
        normal.append(filename)
    covid_img = dict()
    covid_img["client1"] = round(len(covid) * client1)
    covid_img["client2"] = round(len(covid) * client2)
    covid_img["client3"] = round(len(covid) * client3)
    covid_img["client4"] = round(len(covid) * client4)
    print(covid_img)
    normal_img = dict()
    normal_img["client1"] = round(len(normal) * client1)
    normal_img["client2"] = round(len(normal) * client2)
    normal_img["client3"] = round(len(normal) * client3)
    normal_img["client4"] = round(len(normal) * client4)
    print(normal_img)

    s = 0
    i = 0

    for filename in normal:
        i = i+1
        print(i)
        if i <= normal_img["client1"] :
            shutil.copy(input_path + "/normal/" + filename, output_path + "/client1/normal/" + filename)
            s=s+1
            #print("client 1 in process")
        elif i> s and i<= normal_img["client1"]+normal_img["client2"]:
            shutil.copy(input_path + "/normal/" + filename, output_path + "/client2/normal/" + filename)
            s = s + 1
            #print("client 2 in process")

        elif i > s and i <= normal_img["client1"] + normal_img["client2"] + normal_img["client3"]:
            shutil.copy(input_path + "/normal/" + filename, output_path + "/client3/normal/" + filename)
            s = s + 1
            #print("client 3 in process")
            #print("i",i,"s",s)
        elif i >= normal_img["client1"]+normal_img["client2"]+normal_img["client3"] -1 :
            shutil.copy(input_path + "/normal/" + filename, output_path + "/client4/normal/" + filename)
            #print("client 4 in process")

    s = 0
    i = 0
    for filename in covid:
        i = i + 1
        if i <= covid_img["client1"]:
            shutil.copy(input_path + "/covid/" + filename, output_path + "/client1/covid/" + filename)
            s = s + 1
            print()
        elif i > s and i <= covid_img["client1"] + covid_img["client2"]:
            shutil.copy(input_path + "/covid/" + filename, output_path + "/client2/covid/" + filename)
            s = s + 1

        elif i > s and i <= covid_img["client1"] + covid_img["client2"] + covid_img["client3"]:
            shutil.copy(input_path + "/covid/" + filename, output_path + "/client3/covid/" + filename)
            s = s + 1
        elif i >= covid_img["client1"] + covid_img["client2"] + covid_img["client3"] -1 :
            shutil.copy(input_path + "/covid/" + filename, output_path + "/client4/covid/" + filename)


distribute("/Users/macbookair/Desktop/data", "/Users/macbookair/Desktop/fl-dataset", 0.25, 0.25, 0.25, 0.25)