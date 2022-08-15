import numpy as np
import os
import json
import psutil

metaFiles = []
metaDir = ".\\metadata"
newMetaDir = ".\\validMeta"
imagesDir = ".\\images"
n = 337038

_BLACKLIST_ = ['@', '!', '|', '\'', '.', '(', ')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '>', '<', '+', ':', 'breast', 'lowres', 'highres', 'boob', 'translate', 'request', '_no_', '_ni_', 'elemental_hero', 'bra', '_costume', '~', '`', 'harem', 'gangbang', 'ass', 'penis', 'panties', 'cum', 'sex', 'orgasm', 'squirting', 'micro', 'futanari', 'ray', 'drug', 'guro', 'hentai', 'revealing', 'thong', 'cure', 'legs_apart', 'testicles', '/']

_PREPROCESS_ = True

if _PREPROCESS_:

    _PROCESS_JSONS_ =  False

    if _PROCESS_JSONS_:
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
            missedTotal += missed
            print("found", found, "images...")
        print("found ", foundTotal, " files in ", foundTotal + missedTotal, "lookups")
        n = foundTotal

    _GENERATE_TAGS_ = True
    if _GENERATE_TAGS_:

        Lines = []
        processedDataFiles = []
        Tags = []


        for filePath in os.listdir(newMetaDir):
                processedDataFiles.append(os.path.join(newMetaDir, filePath))


        count = 0
        for JSON in processedDataFiles:
            countTotal = 0
            File = open(JSON, 'r', encoding="utf-8")

            Lines = File.readlines()

            for line in Lines:
                count = 0
                metadata = json.loads(line)
                tags = metadata["tags"]
                
                for tag in tags:
                    clean = True
                    for black in _BLACKLIST_:
                        if black in tag['name']:
                            clean = False
                    if tag['name'] not in Tags and clean:
                        Tags.append(tag['name'])
                        count += 1
                        #print("added tag ", tag['name'])
                countTotal += count
                    
            File.close()
            print(JSON, "finished processing, added", countTotal, "tags")

        print("Found", len(Tags), "unique tags, saving to allTagsShort.txt...")

        with open(".\\allTagsShort.txt", 'w', encoding="utf-8") as fp:
            for item in Tags:
                # write each item on a new line
                fp.write("%s\n" % item)
        print("Saved.")


    _GENERATE_Y_ = False

    if _GENERATE_Y_:

        Tags = []
        Lines = []
        processedDataFiles=[]

        tagSave = open(".\\allTagsShort.txt", 'r', encoding="utf-8")
        Lines = tagSave.readlines()
        tagSave.close()

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
                
                metadata = json.loads(line)
                tags = metadata["tags"]
                #row = []
                heldTags = []
                for hasTag in tags:
                    heldTags.append(hasTag['name'])
                tagIndex = 0
                tagCount = 0
                for tag in Tags:
                    if tag in heldTags:
                        #row.append(1)
                        df[count, tagIndex] = 1
                    else:
                        #row.append(0)
                        df[count, tagIndex] = 0
                    tagIndex += 1
                #np.vstack((df, np.asarray(row)))
                count += 1
                RAMUsage = psutil.virtual_memory().percent
                if (RAMUsage*100)%5 == 0 and (RAMUsage - lastRam) >= 0.05:
                    lastRam = RAMUsage
                    print(RAMUsage, "% of RAM used for", count, "lines, current shape:", df.shape)
                    #print(df[:,:])

                if RAMUsage >= 75:
                    print("Memory threshold reached, stopping dataframe generation...")
                    break
                #TODO add row to matrix

            File.close()
            print(JSON, "finished processing, added", count)
        with open('Y.npy', 'wb') as f:
            np.save(f, df)


        ######################## BREAKS, NOT ENOUGH RAM ########################
        #print("calculating independent tags...")
        #_, inds = sympy.Matrix(df).rref()
        #newTags = newTags[inds]
        #print("found", newTags.size(), "indepenent tags, saving...")
        #goodTagSave = open(".\\independentTags.txt", 'w', encoding="utf-8")
        #goodTagSave.writelines(Tags.tolist())
        #goodTagSave.close()
        ########################################################################

        print("Done.")

    _GENERATE_IMAGE_LIST_ = False

    if _GENERATE_IMAGE_LIST_:
        Images = []
        processedDataFiles = []

        for filePath in os.listdir(newMetaDir):
                processedDataFiles.append(os.path.join(newMetaDir, filePath))

        totalCount = 0
        count = 0
        for JSON in processedDataFiles:
            File = open(JSON, 'r', encoding="utf-8")

            totalCount += count
            count = 0

            Lines = File.readlines()
            for line in Lines:

                count += 1
                metadata = json.loads(line)

                ID = metadata["id"]
                imagePath = os.path.join( imagesDir, str( int(ID) % 1000 ).zfill(4), ID + '.jpg' )
                Images.append(imagePath)

            File.close()
            print(JSON, "finished processing, added", count, " images to list")

        print("found", totalCount, "images, saving paths to [Images.txt]...")
        with open(".\\Images.txt", 'w', encoding="utf-8") as fp:
            for path in Images:
                # write each item on a new line
                fp.write("%s\n" % path)

        with open('Images.npy', 'wb') as f:
                np.save(f, Images)

        print("Saved.")

    _FIND_COMMON_ = True

    if _FIND_COMMON_:
        from progress.bar import Bar
        import pandas as pd

        Tags = []
        Lines = []
        processedDataFiles=[]

        tagSave = open(".\\allTagsShort.txt", 'r', encoding="utf-8")
        Lines = tagSave.readlines()
        tagSave.close()

        for line in Lines:
            Tags.append(line.strip())

        print("Starting allocation...")
        newTags = np.asarray(Tags)
        tagCount = pd.DataFrame(np.zeros((1, len(Tags))), columns=newTags)
        print("Allocated array of shape", tagCount.shape)
        count = 1
        for filePath in os.listdir(newMetaDir):
                processedDataFiles.append(os.path.join(newMetaDir, filePath))


        for JSON in processedDataFiles:
            File = open(JSON, 'r', encoding="utf-8")
            contents = File.read()
            File.close()
            message = str(count) + '/17 '
            bar = Bar(message, max=len(Tags), suffix='%(percent).1f%% - %(eta)ds')

            for tag in Tags:
                tagCount[tag][0] += contents.count(tag)
                bar.next()
            bar.finish()

            print("finished", JSON)
            count += 1
        tagCountSorted = tagCount.sort_values(by=0, axis=1)
        print("finished processing common tags, saving...")
        with open(".\\RankedTags.txt", 'w', encoding="utf-8") as fp:
            for tag in tagCountSorted.columns:
                # write each item on a new line
                fp.write("%s\n" % tag)

        tagCountSorted[:, :10000].to_pickle('RankedTags_10k.pd')


        ######################## BREAKS, NOT ENOUGH RAM ########################
        #print("calculating independent tags...")
        #_, inds = sympy.Matrix(df).rref()
        #newTags = newTags[inds]
        #print("found", newTags.size(), "indepenent tags, saving...")
        #goodTagSave = open(".\\independentTags.txt", 'w', encoding="utf-8")
        #goodTagSave.writelines(Tags.tolist())
        #goodTagSave.close()
        ########################################################################

        print("Done.")

    _SPLIT_ = False

    if _SPLIT_:
        print("loading...")
        images = np.load('Images.npy')
        tags_hot = np.load('Y.npy')
    
        ################ TOO MUCH MEMORY #####################
        #from sklearn.model_selection import train_test_split
        #
        #print("splitting...")
        #X_train, X_val, y_train, y_val = train_test_split(
        #    images, tags_hot, test_size=0.2 #, random_state=0 if need reproducable
        #    )
        ################################################

        def shuffle(X, Y, test_proportion):
            ratio = int(X.shape[0]/test_proportion) #should be int
            X_train = X[ratio:]
            X_val =  X[:ratio]
            y_train = Y[ratio:,:]
            y_val =  Y[:ratio,:]
            return X_train, X_val, y_train, y_val

        X_train, X_val, y_train, y_val = shuffle(images, tags_hot, 5)

        print("saving...")
        np.save('./split/X_train.npy', X_train)
        np.save('./split/X_val.npy', X_val)
        np.save('./split/y_train.npy', y_train)
        np.save('./split/y_val.npy', y_val)
        print('train/val split complete')

_TRAIN_ = False

if _TRAIN_:
    
    import keras
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import cv2

    class My_Custom_Generator(keras.utils.Sequence) :
  
        def __init__(self, image_filenames, labels, batch_size) :
            self.image_filenames = image_filenames
            self.labels = labels
            self.batch_size = batch_size
    
    
        def __len__(self) :
            return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
        def __getitem__(self, idx) :
            batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
            batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
            return np.array([
                    cv2.imread(str(file_name))
                        for file_name in batch_x])/255.0, np.array(batch_y)

    
    def makeModel(outSize):

        keras.backend.clear_session()

        inputs = keras.Input(shape=(512,512,3)) #(512, 512, 3)

        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs) # (510, 510, 32)
        x = layers.MaxPooling2D(pool_size=2)(x) # (260, 260, 32)
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x) # (258, 258, 64)
        x = layers.MaxPooling2D(pool_size=2)(x) # (129, 129, 64)
        x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x) # (127, 127, 128)
        x = layers.MaxPooling2D(pool_size=2)(x) # (63, 63, 128)
        x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x) # (61, 61, 128)
        x = layers.MaxPooling2D(pool_size=2)(x) # (30, 30, 128)
        x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x) # (28, 28, 128)
        x = layers.MaxPooling2D(pool_size=2)(x) # (14, 14, 128)

        x = layers.Flatten()(x)
        x = layers.Dropout(0.15)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)

        outputs = layers.Dense(outSize, activation="softmax")(x)

        return keras.Model(inputs=inputs, outputs=outputs)

    print("loading data...")
    X_train = np.load('./split/X_train.npy')
    print('loaded [./split/X_train.npy]...')
    X_val = np.load('./split/X_val.npy')
    print('loaded [./split/X_val.npy]...')
    y_train = np.load('./split/y_train.npy')
    print('loaded [./split/y_train.npy]...')
    y_val = np.load('./split/y_val.npy')
    print('loaded [./split/y_val.npy]...')

    #print(X_train.shape)
    #print(X_train.shape[0])
    #print(X_val.shape)
    #print(X_val.shape[0])
    #print(y_train.shape)
    #print(y_train.shape[0])
    #print(y_train.shape[1])
    #print(y_val.shape)
    #print(y_val.shape[0])
    #print(y_val.shape[1])

    batch_size = 50
    
    my_training_batch_generator = My_Custom_Generator(X_train, y_train, batch_size)
    my_validation_batch_generator = My_Custom_Generator(X_val, y_val, batch_size)
    
    print("loading complete, making model...")
    
    #print(np.array(imread(str(X_train[0]))).shape)

    model = makeModel(y_train.shape[1])
    model.summary()
    #input()
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-1,
        decay_steps=50000,
        decay_rate=0.95)


    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="callback.keras",
        save_best_only=True,
        monitor="val_accuracy")
    ]
    
    
    print("training start")
    
    model.fit_generator(generator=my_training_batch_generator,
                   steps_per_epoch = int(X_train.shape[0] // batch_size),
                   epochs = 50,
                   verbose = 1,
                   validation_data = my_validation_batch_generator,
                   validation_steps = int(X_val.shape[0] // batch_size))
    
    print("saving model")
    
    keras.save('model.kmodel')
    
    print('Done.')