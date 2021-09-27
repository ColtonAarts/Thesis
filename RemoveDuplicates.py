from sklearn.model_selection import train_test_split
import pandas

# in_file = open("DataFile.csv", "r", encoding="utf8").read().split("\n")
#
# out_file = open("noDuplicates.csv", "w+", encoding="utf8")
#
# seen_text = set()
#
# for line in in_file:
#     split = line.split(",")
#     if split[-2] not in seen_text:
#         seen_text.add(split[-2])
#         out_file.write(line + "\n")

df = pandas.read_csv("noDuplicates.csv")

text = df["Tweet"].values
labels = df["Emotion"].values

text_train, text_test, labels_train, labels_test = train_test_split(text, labels, test_size=0.25, random_state=42)

out_train = open("noDuplicates25Train.csv", "w+", encoding="utf8")
out_test = open("noDuplicates25Test.csv", "w+", encoding="utf8")
out_test.write("Tweet,Emotion\n")
for x in range(len(text_test)):
    out_test.write(text_test[x] + "," + labels_test[x] + "\n")
out_train.write("Tweet,Emotion\n")
for x in range(len(text_train)):
    out_train.write(text_train[x] + "," + labels_train[x] + "\n")
