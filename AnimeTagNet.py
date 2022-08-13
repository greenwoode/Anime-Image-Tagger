import numpy as np
import pandas as pd
#import tensorflow as tf
import os
import json
#import sympy
#import psutil

metaFiles = []
metaDir = ".\\metadata"
newMetaDir = ".\\validMeta"
imagesDir = ".\\images"
n = 337038

_PROCESSED_JSONS_ = True

if not _PROCESSED_JSONS_:
    for filePath in os.listdir(metaDir):
        metaFiles.append(os.path.join(metaDir, filePath))

    print("found ", len(metaFiles), " metadata files")

    foundTotal = 0
    missedTotal = 0
    for JSON in metaFiles:

        metaFile = open(JSON, 'r', encoding="utf-8")
        newMetaFile = open(JSON.replace(metaDir, newMetaDir), 'w+', encoding="utf-8")


        found = 0
        missed = 0
        Lines = metaFile.readlines()
        total = len(Lines)
        for line in Lines:

            metadata = json.loads(line)
            ID = metadata["id"]
            imagePath = os.path.join( imagesDir, str( int(ID) % 1000 ).zfill(4), ID + '.jpg' )
        
            if os.path.exists(imagePath):
                #print("(", found + missed, "/", total, "): ",  "found [", imagePath, "]; saving matadata...")
                found += 1
                newMetaFile.write(line)
            else:
                missed += 1

            #if ((found + missed) % 1000) == 0:
                #print(found + missed, "; ",  imagePath)
        metaFile.close()
        newMetaFile.close()

        foundTotal += found
        found  = 0
        missedTotal += missed
        missed = 0

    print("found ", foundTotal, " files in ", foundTotal + missedTotal, "lookups")

_PROCESSED_TAGS_ = True
if not _PROCESSED_TAGS_:
    Lines = []
    processedDataFiles = []
    Tags = []


    for filePath in os.listdir(newMetaDir):
            processedDataFiles.append(os.path.join(newMetaDir, filePath))


    count = 0
    for JSON in processedDataFiles:
        count = 0
        File = open(JSON, 'r', encoding="utf-8")

        Lines = File.readlines()

        for line in Lines:

            metadata = json.loads(line)
            tags = metadata["tags"]

            for tag in tags:
                if tag['name'] not in Tags:
                    Tags.append(tag['name'])
                    count += 1
                    #print("added tag ", tag['name'])

        File.close()
        print(JSON, "finished processing, added", count, "tags")

    print("Found", len(Tags), "unique tags, saving to allTags.txt...")

    with open(".\\allTags.txt", 'w', encoding="utf-8") as fp:
        for item in Tags:
            # write each item on a new line
            fp.write("%s\n" % item)
    print("Saved.")


_REMOVED_DEPENDENT_ = False

if not _REMOVED_DEPENDENT_:

    Tags = []
    Lines = []
    processedDataFiles=[]

    tagSave = open(".\\allTags.txt", 'r', encoding="utf-8")
    Lines = tagSave.readlines()

    for line in Lines:
        Tags.append(line.strip())

    print("Starting allocation...")
    newTags = np.asarray(Tags)
    df = np.empty((n, len(Tags)), dtype=np.int8)
    print("Allocated array of shape", df.shape)
    count = 0
    for filePath in os.listdir(newMetaDir):
            processedDataFiles.append(os.path.join(newMetaDir, filePath))


    for JSON in processedDataFiles:
        File = open(JSON, 'r', encoding="utf-8")

        Lines = File.readlines()
        lastRam = 0
        for line in Lines:
            count += 1
            metadata = json.loads(line)
            tags = metadata["tags"]
            #row = []
            heldTags = []
            for hasTag in tags:
                heldTags.append(hasTag['name'])
            tagIndex = 0
            for tag in Tags:
                if tag in hasTag:
                    #row.append(1)
                    df[count, tagIndex] = 1
                else:
                    #row.append(0)
                    df[count, tagIndex] = 0
                tagIndex += 1
            #np.vstack((df, np.asarray(row)))

            RAMUsage = psutil.virtual_memory().percent
            if (RAMUsage*100)%5 == 0 and (RAMUsage - lastRam) >= 0.05:
                lastRam = RAMUsage
                print(RAMUsage, "% of RAM used for", count, "lines, current shape:", df.shape)
                print(df[:,:])

            if RAMUsage >= 75:
                print("Memory threshold reached, stopping dataframe generation...")
                break
            #TODO add row to matrix

        File.close()
        print(JSON, "finished processing, added", count, "/ 340k rows, resulting in a matrix of shape:", df.shape)
        with open('Y.npy', 'wb') as f:
            np.save(f, df)
    #print("calculating independent tags...")
    #_, inds = sympy.Matrix(df).rref()
    #newTags = newTags[inds]
    #print("found", newTags.size(), "indepenent tags, saving...")
    #goodTagSave = open(".\\independentTags.txt", 'w', encoding="utf-8")
    #goodTagSave.writelines(Tags.tolist())
    #goodTagSave.close()
    print("Done.")