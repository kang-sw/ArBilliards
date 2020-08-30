import cv2 as cv

dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

vimg = None

for i in range(2):
    himg = None
    for j in range(2):
        idx = i * 2 + j
        bdr = 50
        img = cv.aruco.drawMarker(dict, idx, 180, borderBits=1)
        img = cv.copyMakeBorder(img, bdr, bdr, bdr, bdr, cv.BORDER_CONSTANT, value=255)
        img = cv.copyMakeBorder(img, bdr, bdr, bdr, bdr, cv.BORDER_CONSTANT, value=0)

        img = cv.putText(img, '{}'.format(idx), (int(bdr*1.5) - 10, int(bdr*2)), cv.FONT_HERSHEY_PLAIN, 1.3, [0], 1)

        if himg is not None:
            himg = cv.hconcat([himg, img])
        else:
            himg = img

    if vimg is not None:
        vimg = cv.vconcat([vimg, himg])
    else:
        vimg = himg

bdr = 50
vimg = cv.copyMakeBorder(vimg, bdr, bdr, bdr, bdr, cv.BORDER_CONSTANT, dst=vimg)
vimg = [255] - vimg
cv.imwrite('print.png', vimg)
