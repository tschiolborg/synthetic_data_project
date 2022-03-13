import os


annos = os.listdir(os.path.join("data", "MTSD", "annotations"))
print(len(annos))

train = os.listdir(os.path.join("data", "MTSD", "images", "train"))
val = os.listdir(os.path.join("data", "MTSD", "images", "val"))
test = os.listdir(os.path.join("data", "MTSD", "images", "test"))

print(len(train) + len(val) + len(test))
print(len(train) + len(val))
print(len(train))
