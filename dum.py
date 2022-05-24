######################################################################3
########################SOME CODE SNIPPETS THAT MIGHT BE USEFUL##########33
##########3##############################################################

# A sample of the images before preprocessing for comparison.
cmp_old_image = cv2.imread(cmp('female', 4))
icd_old_image = cv2.imread(icd('male', 7))

# Preprocess the images and write them to the disk
image_paths = glob(cmp('*', '*')) + glob(icd('*', '*'))

fig = plt.figure(figsize=(8, 8))
for i, image_path in enumerate(image_paths[0:3]):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Skip this image if it's corrupt, like F87.jpg
    if img is None:
        continue
    i *= 3
    fig.add_subplot(4, 4, i + 1)
    plt.imshow(img)

    blr = cv2.GaussianBlur(img, GAUSSIAN_BLUR_KERNEL_SIZE, 0)
    fig.add_subplot(4, 4, i + 2)
    plt.imshow(blr)

    the = cv2.adaptiveThreshold(blr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    fig.add_subplot(4, 4, i + 3)
    plt.imshow(the)
plt.show()

cmp_new_image = cv2.imread(cmp('female', 4))
icd_new_image = cv2.imread(icd('male', 7))

######## GLCM plot
males = []
females = []

for image_path in map(feat('glcm'), CMP_IMAGES):
    gender = 'male'
    if 'female' in image_path:
        females.append(np.loadtxt(image_path))
    else:
        males.append(np.loadtxt(image_path))
males = np.array(males)
females = np.array(females)

for f1 in range(0, len(males[0])):
    for f2 in range(0, len(males[0])):
        if f1 == f2:
            continue
        #print(f1, f2)
        plt.plot(males[:, f1], males[:, f2], 'bo')
        plt.plot(females[:, f1], females[:, f2], 'rx')
        #plt.show()
        

#### SVM with GLCM features
for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
    for c in range(1, 100000, 1000):
        clf = svm.SVC(kernel=kernel, C=c, probability=True)
        clf.fit(X_train, y_train)
        if c == 40001:
            print(clf.predict(X_test))
        print(clf.score(X_test, y_test), f'{kernel}/C={c}')
#for i, mad in enumerate(y_test):
#    if mad == 'female':
#        print(clf.predict([X_test.iloc[i]]))


##### pandas with glcm
feature_columns = [f'{prop}_{dist}_{angle * 180 / np.pi}'
                   for prop in props + ['entropy']
                   for dist in distances
                   for angle in angles]

#cmp_features = pd.DataFrame(cmp_features)#, columns=feature_columns + ['observation'])
#icd_features = pd.DataFrame(icd_features)#, columns=feature_columns + ['observation'])


##### best c svm
l = []
for c in range(1, 55000, 1000):
    l.append(svm_test(all_features, C=c, times=100, kernel='rbf')[0])
max(l), [c for c in range(1, 55000, 1000)][l.index(max(l))]

##### peak nomalized
for col in range(len(cmp_features[0])):
    f = cmp_features[:,col]
    print(min(f), max(f), sum(f))


##### glcm + lbp train togehter
lbp_features = np.loadtxt(feat('lbp', CMP))
glcm_features = np.loadtxt(feat('glcm', CMP))
#print(len(lbp_features[0]), len(glcm_features[0]))
#print(lbp_features[:,:18])
#print(glcm_features)
features = np.append(lbp_features[:,:18], glcm_features, axis=1)
svm_test(features, C=10, kernel='rbf', test_size=0.2, times=100)


######### append merge 2 datasets together
# np.append(cmp_features, icd_features, axis=0)

######### select specific features from a feature set, note that the last col is the observation
# features = np.append(features[:,12:20], features[:,[-1]], axis=1)


############# old LBP
def ojalat_lbp(image, n_points=16, radius=2):
    # From: IEEE Trans Pattern Anal Mach Intell 24(7):971–987
    # Partial credits: https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    n_uniform_bins = n_points * (n_points - 1) + 2
    n_non_uniform_bins = 1
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, n_uniform_bins + n_non_uniform_bins ),
                           range=(0, n_points + 2))
    return hist

####### used to get best lbp technique, discovered nri_uniform using it
%%time
import random

def ojalat_lbp(image, n_points=16, radius=2):
    lbp = local_binary_pattern(image, n_points, radius, method='nri_uniform')
    # [n * (n - 1) + 2] for uniform bp and [1] more bin for non uniform bp.
    n_bins = n_points * (n_points - 1) + 2 + 1
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_bins + 1))
    return hist

# LBP features.
features = []
random.shuffle(CMP_IMAGES)

for image_path in map(pre('bw'), CMP_IMAGES):
    image = imread_bw(image_path)
    if image is None: continue
    features.append(np.append(ojalat_lbp(image), gender(image_path)))
    
print(len(features), len(features[0]))
features = norm(features)
print(features)

svm_test(features, C=10, kernel='rbf', test_size=0.2, times=100)


########## prune usless features
cmp_features = np.loadtxt(feat('lbp', CMP))
icd_features = np.loadtxt(feat('lbp', ICD))

all_features = np.append(cmp_features, icd_features, axis=0)
zero_cols = all_features[0] == all_features[1]
for feature_v in all_features:
    zero_cols &= all_features[0] == feature
print(~zero_cols)
print(all_features)
print('--------------')
print(all_features[:,~zero_cols])


######### Not needed since they cleaned the dataset
# Remove corrupt images from our image lists, like F87.jpg.
for image_path in ALL_IMAGES:
    if cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) is None:
        ALL_IMAGES.remove(image_path)
        TST_IMAGES.remove(image_path)
        CMP_IMAGES.remove(image_path)
        ICD_IMAGES.remove(image_path)
        

######## prev implementation chain code free man
def freeman_cc_hist(contours):
        DIR8S = {
            (1, 0): 0,
            (1, 1): 1,
            (0, 1): 2,
            (-1, 1): 3,
            (-1, 0): 4,
            (-1, -1): 5,
            (0, -1): 6,
            ( 1, -1): 7,
            }
        # Create an 8-bin histogram.
        hist = collections.defaultdict(int)
        for contour in contours:
            previous_point = contour[0]
            for point in contour[1:]:
                direction = DIR8S[tuple((point-previous_point)[0])]
                hist[direction] += 1
                previous_point = point
        return [hist[direction] for direction in range(8)]